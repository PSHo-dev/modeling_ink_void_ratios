from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MotifSpec:
    name: str
    description: str
    bbox: list[float] | None = None
    role: str | None = None
    must_preserve: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotifSpec:
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            bbox=data.get("bbox"),
            role=data.get("role"),
            must_preserve=list(data.get("must_preserve", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.bbox is not None:
            payload["bbox"] = self.bbox
        if self.role:
            payload["role"] = self.role
        if self.must_preserve:
            payload["must_preserve"] = self.must_preserve
        return payload


@dataclass
class MotifAsset:
    spec: MotifSpec
    crop_path: Path | None
    asset_path: Path
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "crop_path": str(self.crop_path) if self.crop_path else None,
            "asset_path": str(self.asset_path),
            "prompt": self.prompt,
        }


MOTIF_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_summary": {"type": "string"},
        "motifs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short snake_case motif name.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Visual description of the motif to isolate.",
                    },
                    "bbox": {
                        "type": "array",
                        "description": "Normalized [x, y, width, height] bounding box.",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "role": {
                        "type": "string",
                        "description": "Compositional role such as distant mountain, pavilion, figure, mist, foreground.",
                    },
                    "must_preserve": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific details and brushwork that must survive extraction.",
                    },
                },
                "required": ["name", "description", "bbox", "role", "must_preserve"],
            },
        },
    },
    "required": ["source_summary", "motifs"],
}
