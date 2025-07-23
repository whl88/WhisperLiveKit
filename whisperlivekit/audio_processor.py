import asyncio
import numpy as np
from time import time, sleep
import math
import logging
import traceback
from datetime import timedelta
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper_streaming_custom.whisper_online import online_factory
from whisperlivekit.core import TranscriptionEngine
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState

# Set up logging once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self, **kwargs):
        """Initialize the audio processor with configuration, models, and state."""
        
        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)
        
        # Audio processing settings
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.last_ffmpeg_activity = time()
        self.ffmpeg_health_check_interval = 5
        self.ffmpeg_max_idle_time = 10

        # State management
        self.is_stopping = False
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""
        
        # Models and processing
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        self.diarization = models.diarization
        
        self.ffmpeg_manager = FFmpegManager(
            sample_rate=self.sample_rate,
            channels=self.channels
        )
        
        async def handle_ffmpeg_error(error_type: str):
            logger.error(f"FFmpeg error: {error_type}")
            self._ffmpeg_error = error_type
        
        self.ffmpeg_manager.on_error_callback = handle_ffmpeg_error
        self._ffmpeg_error = None
        
        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.pcm_buffer = bytearray()

        # Task references
        self.transcription_task = None
        self.diarization_task = None
        self.ffmpeg_reader_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []
        
        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with new data."""
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """Thread-safe update of diarization with new data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))
            
    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))
                
            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_transcription,
                "remaining_time_diarization": remaining_diarization
            }
            
    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.full_transcription = self.last_response_content = ""
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it."""
        beg = time()
        
        while True:
            try:
                # Check if FFmpeg is running
                state = await self.ffmpeg_manager.get_state()
                if state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot read data")
                    break
                elif state == FFmpegState.STOPPED:
                    logger.info("FFmpeg is stopped")
                    break
                elif state != FFmpegState.RUNNING:
                    logger.warning(f"FFmpeg is in {state} state, waiting...")
                    await asyncio.sleep(0.5)
                    continue
                
                current_time = time()
                elapsed_time = math.floor((current_time - beg) * 10) / 10
                buffer_size = max(int(32000 * elapsed_time), 4096)
                beg = current_time

                chunk = await self.ffmpeg_manager.read_data(buffer_size)
                        
                if not chunk:
                    if self.is_stopping:
                        logger.info("FFmpeg stdout closed, stopping.")
                        break
                    else:
                        # No data available, but not stopping - FFmpeg might be restarting
                        await asyncio.sleep(0.1)
                        continue
                    
                self.pcm_buffer.extend(chunk)
                        
                # Send to diarization if enabled
                if self.args.diarization and self.diarization_queue:
                    await self.diarization_queue.put(
                        self.convert_pcm_to_float(self.pcm_buffer).copy()
                    )

                # Process when enough data
                if len(self.pcm_buffer) >= self.bytes_per_sec:
                    if len(self.pcm_buffer) > self.max_bytes_per_sec:
                        logger.warning(
                            f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                            f"Consider using a smaller model."
                        )

                    # Process audio chunk
                    pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:self.max_bytes_per_sec])
                    self.pcm_buffer = self.pcm_buffer[self.max_bytes_per_sec:]
                    
                    # Send to transcription if enabled
                    if self.args.transcription and self.transcription_queue:
                        await self.transcription_queue.put(pcm_array.copy())
                    
                    # Sleep if no processing is happening
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                # Try to recover by waiting a bit
                await asyncio.sleep(1)
                
                # Check if we should exit
                if self.is_stopping:
                    break
        
        logger.info("FFmpeg stdout processing finished. Signaling downstream processors.")
        if self.args.transcription and self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)
            logger.debug("Sentinel put into transcription_queue.")
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(SENTINEL)
            logger.debug("Sentinel put into diarization_queue.")


    async def transcription_processor(self):
        """Process audio chunks for transcription."""
        self.full_transcription = ""
        self.sep = self.online.asr.sep
        cumulative_pcm_duration_stream_time = 0.0
        
        while True:
            try:
                pcm_array = await self.transcription_queue.get()
                if pcm_array is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    self.transcription_queue.task_done()
                    break
                
                if not self.online: # Should not happen if queue is used
                    logger.warning("Transcription processor: self.online not initialized.")
                    self.transcription_queue.task_done()
                    continue

                asr_internal_buffer_duration_s = len(getattr(self.online, 'audio_buffer', [])) / self.online.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.end_buffer)

                logger.info(
                    f"ASR processing: internal_buffer={asr_internal_buffer_duration_s:.2f}s, "
                    f"lag={transcription_lag_s:.2f}s."
                )
                
                # Process transcription
                duration_this_chunk = len(pcm_array) / self.sample_rate if isinstance(pcm_array, np.ndarray) else 0
                cumulative_pcm_duration_stream_time += duration_this_chunk
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time

                self.online.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                new_tokens, current_audio_processed_upto = self.online.process_iter()
                
                # Get buffer information
                _buffer_transcript_obj = self.online.get_buffer()
                buffer_text = _buffer_transcript_obj.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    self.full_transcription += validated_text
                    
                    if buffer_text.startswith(validated_text):
                        buffer_text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript_obj.end is not None:
                    candidate_end_times.append(_buffer_transcript_obj.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                new_end_buffer = max(candidate_end_times)
                
                await self.update_transcription(
                    new_tokens, buffer_text, new_end_buffer, self.full_transcription, self.sep
                )
                self.transcription_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL : # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        logger.info("Transcription processor task finished.")


    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization."""
        buffer_diarization = ""
        
        while True:
            try:
                pcm_array = await self.diarization_queue.get()
                if pcm_array is SENTINEL:
                    logger.debug("Diarization processor received sentinel. Finishing.")
                    self.diarization_queue.task_done()
                    break
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                
                async with self.lock:
                    new_end = diarization_obj.assign_speakers_to_tokens(
                        self.end_attributed_speaker, 
                        self.tokens,
                        use_punctuation_split=self.args.punctuation_split
                    )
                    self.end_attributed_speaker = new_end
                    if buffer_diarization:
                        self.buffer_diarization = buffer_diarization
                
                self.diarization_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL:
                    self.diarization_queue.task_done()
        logger.info("Diarization processor task finished.")


    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                ffmpeg_state = await self.ffmpeg_manager.get_state()
                if ffmpeg_state == FFmpegState.FAILED and self._ffmpeg_error:
                    yield {
                        "status": "error",
                        "error": f"FFmpeg error: {self._ffmpeg_error}",
                        "lines": [],
                        "buffer_transcription": "",
                        "buffer_diarization": "",
                        "remaining_time_transcription": 0,
                        "remaining_time_diarization": 0
                    }
                    self._ffmpeg_error = None
                    await asyncio.sleep(1)
                    continue
                
                # Get current state
                state = await self.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                sep = state["sep"]
                
                # Add dummy tokens if needed
                if (not tokens or tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.add_dummy_token()
                    sleep(0.5)
                    state = await self.get_current_state()
                    tokens = state["tokens"]
                
                # Format output
                previous_speaker = -1
                lines = []
                last_end_diarized = 0
                undiarized_text = []
                
                # Process each token
                for token in tokens:
                    speaker = token.speaker
                    
                    # Handle diarization
                    if self.args.diarization:
                        if (speaker in [-1, 0]) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)
                            continue
                        elif (speaker in [-1, 0]) and token.end < end_attributed_speaker:
                            speaker = previous_speaker
                        if speaker not in [-1, 0]:
                            last_end_diarized = max(token.end, last_end_diarized)

                    # Group by speaker
                    if speaker != previous_speaker or not lines:
                        lines.append({
                            "speaker": speaker,
                            "text": token.text,
                            "beg": format_time(token.start),
                            "end": format_time(token.end),
                            "diff": round(token.end - last_end_diarized, 2)
                        })
                        previous_speaker = speaker
                    elif token.text:  # Only append if text isn't empty
                        lines[-1]["text"] += sep + token.text
                        lines[-1]["end"] = format_time(token.end)
                        lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
                
                # Handle undiarized text
                if undiarized_text:
                    combined = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined += sep
                    await self.update_diarization(end_attributed_speaker, combined)
                    buffer_diarization = combined
                
                response_status = "active_transcription"
                final_lines_for_response = lines.copy()

                if not tokens and not buffer_transcription and not buffer_diarization:
                    response_status = "no_audio_detected"
                    final_lines_for_response = []
                elif response_status == "active_transcription" and not final_lines_for_response:
                    final_lines_for_response = [{
                        "speaker": 1,
                        "text": "",
                        "beg": format_time(state.get("end_buffer", 0)),
                        "end": format_time(state.get("end_buffer", 0)),
                        "diff": 0
                    }]
                
                response = {
                    "status": response_status,
                    "lines": final_lines_for_response,
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "remaining_time_transcription": state["remaining_time_transcription"],
                    "remaining_time_diarization": state["remaining_time_diarization"]
                }
                
                current_response_signature = f"{response_status} | " + \
                                           ' '.join([f"{line['speaker']} {line['text']}" for line in final_lines_for_response]) + \
                                           f" | {buffer_transcription} | {buffer_diarization}"
                
                if current_response_signature != self.last_response_content and \
                   (final_lines_for_response or buffer_transcription or buffer_diarization or response_status == "no_audio_detected"):
                    yield response
                    self.last_response_content = current_response_signature
                
                # Check for termination condition
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False
                    
                    if all_processors_done:
                        logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                        final_state = await self.get_current_state()
                        return
                
                await asyncio.sleep(0.1)  # Avoid overwhelming the client
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Back off on error
        
    async def create_tasks(self):
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        success = await self.ffmpeg_manager.start()
        if not success:
            logger.error("Failed to start FFmpeg manager")
            async def error_generator():
                yield {
                    "status": "error", 
                    "error": "FFmpeg failed to start. Please check that FFmpeg is installed.",
                    "lines": [],
                    "buffer_transcription": "",
                    "buffer_diarization": "",
                    "remaining_time_transcription": 0,
                    "remaining_time_diarization": 0
                }
            return error_generator()

        if self.args.transcription and self.online:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)
            
        if self.args.diarization and self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor(self.diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)
        
        self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
        self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
        processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)
        
        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(10)
                
                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                
                # Check FFmpeg status through the manager
                ffmpeg_state = await self.ffmpeg_manager.get_state()
                if ffmpeg_state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, notifying results formatter")
                    # FFmpeg manager will handle its own recovery
                elif ffmpeg_state == FFmpegState.STOPPED and not self.is_stopping:
                    logger.warning("FFmpeg unexpectedly stopped, attempting restart")
                    await self.ffmpeg_manager.restart()
                    
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")        
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
        
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")
        await self.ffmpeg_manager.stop()
        logger.info("FFmpeg manager stopped.")
        if self.args.diarization and hasattr(self, 'diarization') and hasattr(self.diarization, 'close'):
            self.diarization.close()
        logger.info("AudioProcessor cleanup complete.")


    async def process_audio(self, message):
        if not message:
            self.is_stopping = True
            return

        if self.is_stopping:
            return

        # 直接累加PCM数据
        self.pcm_buffer.extend(message)

        # 达到一秒音频长度时处理
        while len(self.pcm_buffer) >= self.bytes_per_sec:
            chunk = self.pcm_buffer[:self.bytes_per_sec]
            self.pcm_buffer = self.pcm_buffer[self.bytes_per_sec:]
            pcm_array = self.convert_pcm_to_float(chunk)
            if self.args.transcription and self.transcription_queue:
                await self.transcription_queue.put(pcm_array.copy())
            if self.args.diarization and self.diarization_queue:
                await self.diarization_queue.put(pcm_array.copy())
