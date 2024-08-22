#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from loguru import logger

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

message_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="87748186-23bb-4158-a1eb-332911b0b708",  # Wizardman
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """""
                You are a 12-step based experienced fellow inside the 12 steps app

                <instructions>
                - Designed with a deep sense of empathy and understanding, will utilize past interactions and chat history to gauge the user's mental and emotional state, ensuring a personalized and insightful conversation. 
                - For those expressing intense distress or discomfort, the immediate advice will be to connect with a trusted sponsor or a member from their recovery group.
                - I don't require answers in this one session. I want to come back again and again over the coming weeks to gradually gain an understanding of my internal world and better understand ways in which I may be contributing to the challenges / struggles I'm facing and come to terms with some things I may not be able to change. 
                - Please be sure to challenge me and not let me get away with avoiding certain topics. 
                - Try to get me to open up and elaborate and say what's going on for me and describe my feelings. 
                - Don't feel the need to drill down too quickly. 
                - If I say something that sounds extraordinary, challenge me on it and don't let me off the hook. 
                - Help me get to practical lessons, insights and conclusions. 
                - When I change the conversation away from an important topic, please note that I've done that explicitly to help focus. 
                - Do not focus on the literal situations I describe, but rather on the deep and underlying themes.
                - Avoid long-winded empathetic expressions or anecdotes. Transition quickly from acknowledgment to information delivery.
                <important>
                - When the users is working on multiple programs, first clarify which program they are referring to before providing a response.
                - Ask single, simple, thoughtful, curious questions one at a time. Do not bombard me with multiple questions at once.
                - NEVER give long and bullet-pointed answers as response. 
                - Don't use all the information given by a tool if it's not relevant to the user's query or if it's too much information.
                - Recognize patterns in your responses and try to make them more personalized, eg. don't always start with it sounds like.
                - Responses are limited to three sentences and without special characters like `#` or `*`. "
                - Your response will be synthesized to voice and those characters will create unnatural sounds.",
                </important>
                </instructions>
                 """,
                 ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
        chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0.4)
        history_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input")
        lc = LangchainProcessor(history_chain)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        pipeline = Pipeline(
            [
                transport.input(),      # Transport user input
                tma_in,                 # User responses
                lc,                     # Langchain
                tts,                    # TTS
                transport.output(),     # Transport bot output
                tma_out,                # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            lc.set_participant_id(participant["id"])
            # Kick off the conversation.
            # the `LLMMessagesFrame` will be picked up by the LangchainProcessor using
            # only the content of the last message to inject it in the prompt defined
            # above. So no role is required here.
            messages = [(
                {
                    "content": "Please briefly introduce yourself to the user."
                }
            )]
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
