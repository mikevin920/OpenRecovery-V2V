#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger

try:
    import pyaudio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use local audio, you need to `pip install pipecat-ai[local]`. On MacOS, you also need to `brew install portaudio`.")
    raise Exception(f"Missing module: {e}")


class LocalAudioInputTransport(BaseInputTransport):

    def __init__(self, py_audio: pyaudio.PyAudio, params: TransportParams):
        super().__init__(params)

        sample_rate = self._params.audio_in_sample_rate
        num_frames = int(sample_rate / 100)  # 10ms of audio

        self._in_stream = py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=params.audio_in_channels,
            rate=params.audio_in_sample_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._in_stream.start_stream()

    async def stop(self):
        await super().stop()
        self._in_stream.stop_stream()

    async def cleanup(self):
        # This is not very pretty (taken from PyAudio docs).
        while self._in_stream.is_active():
            await asyncio.sleep(0.1)
        self._in_stream.close()

        await super().cleanup()

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        if not self._running:
            return (None, pyaudio.paAbort)

        frame = AudioRawFrame(audio=in_data,
                              sample_rate=self._params.audio_in_sample_rate,
                              num_channels=self._params.audio_in_channels)
        self.push_audio_frame(frame)

        return (None, pyaudio.paContinue)


class LocalAudioOutputTransport(BaseOutputTransport):

    def __init__(self, py_audio: pyaudio.PyAudio, params: TransportParams):
        super().__init__(params)

        self._out_stream = py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=params.audio_out_channels,
            rate=params.audio_out_sample_rate,
            output=True)

    def write_raw_audio_frames(self, frames: bytes):
        self._out_stream.write(frames)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._out_stream.start_stream()

    async def stop(self):
        await super().stop()
        self._out_stream.stop_stream()

    async def cleanup(self):
        # This is not very pretty (taken from PyAudio docs).
        while self._out_stream.is_active():
            await asyncio.sleep(0.1)
        self._out_stream.close()

        await super().cleanup()


class LocalAudioTransport(BaseTransport):

    def __init__(self, params: TransportParams):
        self._params = params
        self._pyaudio = pyaudio.PyAudio()

        self._input: LocalAudioInputTransport | None = None
        self._output: LocalAudioOutputTransport | None = None

    #
    # BaseTransport
    #

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = LocalAudioInputTransport(self._pyaudio, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = LocalAudioOutputTransport(self._pyaudio, self._params)
        return self._output
