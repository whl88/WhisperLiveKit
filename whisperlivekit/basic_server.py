from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import WhisperLiveKit, get_parsed_args
from whisperlivekit.audio_processor import AudioProcessor

import asyncio
import logging
import os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    kit = WhisperLiveKit()
    app.state.kit = kit
    logger.info(f"Audio Input mode: {kit.args.audio_input}")

    audio_processor = AudioProcessor()
    app.state.audio_processor = audio_processor
    app.state.results_generator = None # Initialize

    if kit.args.audio_input == "pyaudiowpatch":
        logger.info("Starting PyAudioWPatch processing tasks...")
        try:
            app.state.results_generator = await audio_processor.create_tasks()
        except Exception as e:
             logger.critical(f"Failed to start PyAudioWPatch processing: {e}", exc_info=True)
    else:
        logger.info("WebSocket input mode selected. Processing will start on client connection.")

    yield

    logger.info("Shutting down...")
    if hasattr(app.state, 'audio_processor') and app.state.audio_processor:
        logger.info("Cleaning up AudioProcessor...")
        await app.state.audio_processor.cleanup()
    logger.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(app.state.kit.web_interface())


async def handle_websocket_results(websocket: WebSocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response)
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    audio_processor = app.state.audio_processor
    kit_args = app.state.kit.args
    results_generator = None
    websocket_task = None
    receive_task = None

    try:
        if kit_args.audio_input == "websocket":
            logger.info("WebSocket mode: Starting processing tasks for this connection.")
            results_generator = await audio_processor.create_tasks()
            websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

            async def receive_audio():
                try:
                    while True:
                        message = await websocket.receive_bytes()
                        await audio_processor.process_audio(message)
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected by client (receive_audio).")
                except Exception as e:
                    logger.error(f"Error receiving audio: {e}", exc_info=True)
                finally:
                     logger.debug("Receive audio task finished.")


            receive_task = asyncio.create_task(receive_audio())
            done, pending = await asyncio.wait(
                {websocket_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel() # Cancel the other task

        elif kit_args.audio_input == "pyaudiowpatch":
            logger.info("PyAudioWPatch mode: Streaming existing results.")
            results_generator = app.state.results_generator
            if results_generator is None:
                 logger.error("PyAudioWPatch results generator not available. Was startup successful?")
                 await websocket.close(code=1011, reason="Server error: Audio processing not started.")
                 return

            websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))
            await websocket_task

        else:
             logger.error(f"Unsupported audio input mode configured: {kit_args.audio_input}")
             await websocket.close(code=1011, reason="Server configuration error.")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}", exc_info=True)
        # Attempt to close gracefully
        try:
            await websocket.close(code=1011, reason=f"Server error: {e}")
        except Exception:
            pass # Ignore errors during close after another error
    finally:
        logger.info("Cleaning up WebSocket connection...")
        if websocket_task and not websocket_task.done():
            websocket_task.cancel()
        if receive_task and not receive_task.done():
            receive_task.cancel()

        if kit_args.audio_input == "websocket":
             pass

        logger.info("WebSocket connection closed.")

def main():
    """Entry point for the CLI command."""
    import uvicorn

    # Get the globally parsed arguments
    args = get_parsed_args()

    # Set logger level based on args
    log_level_name = args.log_level.upper()
    # Ensure the level name is valid for the logging module
    numeric_level = getattr(logging, log_level_name, None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Invalid log level: {args.log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    logging.getLogger().setLevel(numeric_level) # Set root logger level
    # Set our specific logger level too
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to: {log_level_name}")

    # Determine uvicorn log level (map CRITICAL to critical, etc.)
    uvicorn_log_level = log_level_name.lower()
    if uvicorn_log_level == "debug": # Uvicorn uses 'trace' for more verbose than debug
         uvicorn_log_level = "trace"


    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port,
        "reload": False,
        "log_level": uvicorn_log_level,
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }


    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
