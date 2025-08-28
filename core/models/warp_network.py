import numpy as np
import torch
from ..utils.load_model import load_model


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "WarpingNetwork",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device
        self.closed = False  # <-- add

    def __call__(self, feature_3d, kp_source, kp_driving):
        """
        feature_3d: np.ndarray, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        """
        if self.model_type == "onnx":
            pred = self.model.run(None, {"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})
            self.model.infer()
            pred = self.model.buffer["out"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                pred = self.model(
                    torch.from_numpy(feature_3d).to(self.device), 
                    torch.from_numpy(kp_source).to(self.device), 
                    torch.from_numpy(kp_driving).to(self.device)
                ).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred

    def close(self):
        if self.closed:
            return
        try:
            m = self.model
            if m is None:
                return
            if self.model_type == "pytorch":
                try:
                    import torch
                    with torch.no_grad():
                        to = getattr(m, "to", None)
                        if callable(to):
                            to("cpu")
                except Exception:
                    pass
            elif self.model_type == "tensorrt":
                # common TRT handles on custom wrappers
                for attr in ("context", "engine", "runtime", "stream"):
                    obj = getattr(m, attr, None)
                    if hasattr(obj, "destroy"):
                        try: obj.destroy()
                        except Exception: pass
                for meth in ("close", "destroy", "cleanup", "deinit"):
                    fn = getattr(m, meth, None)
                    if callable(fn):
                        try: fn()
                        except Exception: pass
            elif self.model_type == "onnx":
                # ORT sessions don't expose public close; dropping refs + GC is the path.
                pass
        finally:
            self.model = None
            self.closed = True
            # best-effort GC + CUDA cache
            try:
                import gc; gc.collect()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            except Exception:
                pass

    def __del__(self):
        # safety net
        try:
            self.close()
        except Exception:
            pass