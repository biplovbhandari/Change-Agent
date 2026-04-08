"""
ADK bridge for Change-Agent — multi-turn agent using Google ADK.

Tools:
    1) set_image_paths(image_a_path, image_b_path)
    2) detect_changes(image_a_path?, image_b_path?, output_path?)
    3) get_last_statistics()
    4) save_road_change_map(save_path)
"""

from __future__ import annotations
import asyncio, inspect, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
load_dotenv()  # fallback to process working dir

from google.genai import types

# --- repo-relative import for predict.py ---
CURRENT_DIR = Path(__file__).resolve().parent         # .../src/adk_app
SRC_DIR = CURRENT_DIR.parent                           # .../src
import sys
sys.path.append(str(SRC_DIR))
from predict import Change_Perception

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner


# -------------------------
# Model wrapper (stateful)
# -------------------------
@dataclass
class CDArgs:
    data_folder: str = "./data"
    list_path: str = os.environ.get("VOCAB_PATH", "./data/LEVIR_MCI")
    vocab_file: str = "vocab"
    max_length: int = 41
    gpu_id: int = int(os.environ.get("GPU_ID", "-1"))
    checkpoint: str = os.environ.get("MODEL_CHECKPOINT", "./models_ckpt/MCI_model.pth")
    network: str = "segformer-mit_b1"
    encoder_dim: int = 512
    feat_size: int = 16
    dropout: float = 0.1
    n_heads: int = 8
    n_layers: int = 3
    decoder_n_layers: int = 1
    feature_dim: int = 512


class ChangeEngine:
    """Wraps Change_Perception with convenience + memory."""
    def __init__(self):
        args = CDArgs()
        self.model = Change_Perception(args)
        self.image_a_path: Optional[str] = None
        self.image_b_path: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def set_image_paths(self, image_a_path: str, image_b_path: str) -> Dict[str, Any]:
        self.image_a_path = image_a_path
        self.image_b_path = image_b_path
        return {"status": "ok", "image_a_path": image_a_path, "image_b_path": image_b_path}

    def _ensure_paths(self, a: Optional[str], b: Optional[str]):
        a = a or self.image_a_path
        b = b or self.image_b_path
        if not a or not b:
            raise ValueError("Image paths are not set. Call set_image_paths() or pass arguments.")
        return a, b

    def detect(self, image_a_path: Optional[str], image_b_path: Optional[str],
               output_path: Optional[str]) -> Dict[str, Any]:
        a, b = self._ensure_paths(image_a_path, image_b_path)
        if output_path is None:
            stem_a = Path(a).stem
            stem_b = Path(b).stem
            output_path = str(Path("tmp_uploads") / f"{stem_a}__{stem_b}__change_mask.png")
        os.makedirs(Path(output_path).parent, exist_ok=True)

        caption = self.model.generate_change_caption(a, b)
        mask = self.model.change_detection(a, b, output_path)
        road_count = self.model.compute_object_num(mask, "road")
        building_count = self.model.compute_object_num(mask, "building")

        self.last_result = {
            "mask": mask,
            "caption": caption,
            "road_count": road_count,
            "building_count": building_count,
            "image_a_path": a,
            "image_b_path": b,
            "mask_path": output_path,
        }
        return {
            "success": True,
            "caption": caption,
            "mask_path": output_path,
            "statistics": {"roads": road_count, "buildings": building_count},
            "text": f"Change detected: {caption}\n\nStatistics:\n- Roads: {road_count}\n- Buildings: {building_count}",
        }

    def stats(self) -> Dict[str, Any]:
        if not self.last_result:
            return {"success": False, "text": "No previous detection available."}
        return {
            "success": True,
            "statistics": {
                "roads": self.last_result["road_count"],
                "buildings": self.last_result["building_count"],
            },
            "mask_path": self.last_result.get("mask_path"),
            "text": f"{self.last_result['road_count']} roads and {self.last_result['building_count']} buildings have changed.",
        }

    def save_road_only(self, save_path: str) -> Dict[str, Any]:
        if not self.last_result:
            return {"success": False, "text": "Run change detection first."}
        import numpy as np, cv2  # local use
        mask = self.last_result["mask"]
        pred_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        pred_rgb[mask == 1] = [0, 255, 255]  # roads (yellow)
        os.makedirs(Path(save_path).parent, exist_ok=True)
        cv2.imwrite(save_path, pred_rgb)
        return {"success": True, "saved_to": save_path, "text": f"Road change map saved to {save_path}"}


# -------------------------------------
# ADK Agent wrapper & exposed functions
# -------------------------------------
class ADKChangeAgent:
    """
    Small wrapper around ADK's Agent + InMemoryRunner for Streamlit apps.
    Keeps a fixed user_id/session_id so conversations are multi-turn.
    """

    def __init__(
        self,
        model_name: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        user_id: str = "local-user",
        session_id: str = "change-session",
    ):
        self.engine = ChangeEngine()
        self.user_id = user_id
        self.session_id = session_id

        def set_image_paths(image_a_path: str, image_b_path: str) -> dict:
            """Registers (and remembers) the two input image paths for later turns."""
            return self.engine.set_image_paths(image_a_path, image_b_path)

        def detect_changes(
            image_a_path: Optional[str] = None,
            image_b_path: Optional[str] = None,
            output_path: Optional[str] = None,
        ) -> dict:
            """Runs change detection. If paths are omitted, uses the last registered images."""
            return self.engine.detect(image_a_path, image_b_path, output_path)

        def get_last_statistics() -> dict:
            """Returns counts & mask path from the most recent detection."""
            return self.engine.stats()

        def save_road_change_map(save_path: str) -> dict:
            """Saves a PNG with only the road (class=1) changes highlighted."""
            return self.engine.save_road_only(save_path)


        self.agent = Agent(
            name="change_detection_agent",
            model=model_name,
            description="Analyzes pairs of aerial images, detects changes, and answers follow-ups.",
            instruction=(
                "You are a helpful visual change-detection assistant. "
                "When the user asks about changes, call detect_changes. "
                "If they ask counts, call get_last_statistics. "
                "If they mention saving a road map, call save_road_change_map. "
                "If the image paths are provided in conversation or the UI, call set_image_paths once."
            ),
            tools=[set_image_paths, detect_changes, get_last_statistics, save_road_change_map],
        )

        self.runner = InMemoryRunner(self.agent, app_name="Change-Agent")
        self.ensure_session()

    def ensure_session(self):
        """
        Create (or fetch) a session for (user_id, session_id) on the runner's session_service.
        Handles both sync & async create_session() variants and updates self.session_id if the
        service returns a generated ID.
        """
        svc = getattr(self.runner, "session_service", None)
        if not svc:
            return

        # Try create_session with best-effort kwargs for both in-memory & vertex services.
        create = getattr(svc, "create_session", None)
        if not create:
            return

        kwargs = {}
        # Provide app_name if the service expects it
        if hasattr(self.runner, "app_name"):
            kwargs["app_name"] = self.runner.app_name
        # Always include user_id
        kwargs["user_id"] = self.user_id

        # Some services accept an explicit session_id; if so, pass ours
        try:
            if "session_id" in inspect.signature(create).parameters:
                kwargs["session_id"] = self.session_id
        except (TypeError, ValueError):
            pass  # signature inspection can fail on some wrappers; ignore

        # Call sync or async
        try:
            result = create(**kwargs)
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
        except Exception:
            # Try a couple of common alternates
            for alt in ("get_or_create_session", "start_session", "new_session"):
                if hasattr(svc, alt):
                    fn = getattr(svc, alt)
                    try:
                        result = fn(**kwargs)
                        if inspect.iscoroutine(result):
                            result = asyncio.run(result)
                        break
                    except Exception:
                        continue
            else:
                return  # give up quietly; chat() will retry if needed

        # If a session object came back with a generated ID, adopt it
        sid = getattr(result, "id", None)
        if not sid and isinstance(result, dict):
            sid = result.get("id")
        if sid:
            self.session_id = sid

    # --- Helper (ui can call directly) ---
    def register_images(self, a: str, b: str):
        self.engine.set_image_paths(a, b)

    # --- Single multi-turn step ---
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Run one step and return {'response': str, 'actions': [..]}"""
        actions: List[Dict[str, Any]] = []
        final_text_parts: List[str] = []

        # Build ADK Content instead of passing a raw string
        user_content = types.Content(role="user", parts=[types.Part(text=user_message)])

        def _consume_events():
            for event in self.runner.run(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=user_content,   # <-- Content, not str
            ):
                # capture tool calls
                for fc in event.get_function_calls() or []:
                    actions.append({
                        "type": fc.name,
                        "thought": f"Calling tool '{fc.name}'",
                        "args": fc.args or {},
                    })
                # capture tool results
                for fr in event.get_function_responses() or []:
                    result = fr.response or {}
                    for a in reversed(actions):
                        if a["type"] == fr.name and "result" not in a:
                            a["result"] = result
                            break
                    else:
                        actions.append({"type": fr.name, "thought": "Tool result", "result": result})

                # final model text
                if getattr(event, "is_final_response", lambda: False)():
                    parts = getattr(event, "content", None)
                    if parts and getattr(parts, "parts", None):
                        for p in parts.parts:
                            t = getattr(p, "text", None)
                            if t:
                                final_text_parts.append(t)

        # Try once; if the session vanished, (re)create and retry.
        try:
            _consume_events()
        except ValueError as e:
            if "Session not found" in str(e):
                self.ensure_session()
                _consume_events()
            else:
                raise

        final_text = "\n".join(final_text_parts).strip() or "(no text)"
        return {"response": final_text, "actions": actions}
