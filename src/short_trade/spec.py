"""catalog/rules/*.yaml と catalog/common.yaml の読み込みと検証（FR-C05〜C07）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ast
import copy
import re

import yaml

_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def _eval_param_expr(expr: str, params: dict[str, Any]) -> Any:
    """`{{ name }}` や `{{ skip_days + lookback_days }}` の中身を数値として評価する（四則演算のみ）。"""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in params:
                raise SpecError(f"未定義のパラメータ {node.id!r}（params_to_optimize に無い）")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
                                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)):
            raise SpecError(f"パラメータ式に使えない要素です: {expr!r}")
    return eval(compile(tree, "<param>", "eval"), {"__builtins__": {}}, dict(params))


def _format_number(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(v)


def substitute_params(obj: Any, params: dict[str, Any]) -> Any:
    """raw 内のすべての文字列で `{{...}}` を置換する。params_to_optimize の宣言以外は触れない。"""
    if isinstance(obj, str):
        return _PLACEHOLDER.sub(lambda m: _format_number(_eval_param_expr(m.group(1), params)), obj)
    if isinstance(obj, dict):
        return {k: (v if k == "params_to_optimize" else substitute_params(v, params)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_params(v, params) for v in obj]
    return obj


class SpecError(ValueError):
    """仕様ファイルの不備。起動時に検出して落とす（RM-062 と同じ思想）。"""


@dataclass
class StrategySpec:
    id: str
    name: str
    factor: str
    phase: int | None
    common: dict[str, Any]
    raw: dict[str, Any]
    template: dict[str, Any] | None = None       # `{{...}}` を残した原本。with_params で再置換する
    param_values: dict[str, Any] | None = None   # 現在の raw に埋め込まれているパラメータ値

    def with_params(self, **overrides: Any) -> "StrategySpec":
        """宣言済みパラメータを差し替えた新しい仕様を返す（学習／検証の探索用）。"""
        base = self.template if self.template is not None else self.raw
        params = dict(self.params)
        unknown = set(overrides) - set(params)
        if unknown:
            raise SpecError(f"{self.id}: 宣言されていないパラメータ: {sorted(unknown)}（自由度の枠外）")
        params.update(overrides)
        new = StrategySpec(id=self.id, name=self.name, factor=self.factor, phase=self.phase,
                           common=self.common, raw=substitute_params(copy.deepcopy(base), params),
                           template=base, param_values=params)
        _ = new.rank
        return new

    # --- 便利アクセサ ---
    @property
    def entry_conditions(self) -> list[str]:
        return list(self.raw.get("entry", {}).get("conditions", []))

    @property
    def regime_conditions(self) -> list[str]:
        own = list(self.raw.get("regime_filter", []) or [])
        cond = self.common.get("regime", {}).get("market_filter", {}).get("condition")
        common = [cond] if cond else []
        # 個別仕様が同じ条件を明示していても重複評価しないよう集約する
        return list(dict.fromkeys(own + common))

    @property
    def stop_initial(self) -> str:
        return self.raw["exit"]["stop"]["initial"]

    @property
    def stop_trailing(self) -> str | None:
        return self.raw["exit"]["stop"].get("trailing")

    @property
    def stop_ratchet(self) -> bool:
        return bool(self.raw["exit"]["stop"].get("ratchet", False))

    @property
    def exit_rules(self) -> list[dict[str, Any]]:
        return list(self.raw.get("exit", {}).get("rules", []))

    @property
    def risk_pct(self) -> float:
        sizing = self.raw.get("sizing") or {}
        if "risk_pct" in sizing:
            return float(sizing["risk_pct"])
        return float(self.common["risk"]["risk_pct_default"])

    @property
    def definitions(self) -> dict[str, Any]:
        """`definitions:` のうち、式または定数として使えるものを返す。

        入れ子の辞書（ST-04 の vcp など）は専用の検出器が必要であり、式では表せない。
        """
        raw = self.raw.get("definitions") or {}
        return {k: v for k, v in raw.items()
                if isinstance(v, (str, int, float, bool))}

    @property
    def detector_definitions(self) -> dict[str, dict]:
        """`{detector: 名前, ...引数}` 形式の定義。detectors.py の登録名で解決する。"""
        raw = self.raw.get("definitions") or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict) and "detector" in v}

    @property
    def unsupported_definitions(self) -> list[str]:
        """専用の実装が必要だが detector 指定の無い定義。"""
        raw = self.raw.get("definitions") or {}
        return [k for k, v in raw.items() if isinstance(v, dict) and "detector" not in v]

    @property
    def rank(self) -> tuple[str, bool] | None:
        """候補の優先順位: (式, 降順か)。説明文のままなら仕様不備として落とす。"""
        r = self.raw.get("entry", {}).get("rank")
        if r is None:
            return None
        if not isinstance(r, dict) or "by" not in r:
            raise SpecError(f"{self.id}: entry.rank は {{by: 式, order: desc|asc}} の形で書いてください: {r!r}")
        return str(r["by"]), str(r.get("order", "desc")).lower() != "asc"

    @property
    def max_positions(self) -> int | None:
        """戦略固有の同時保有上限（ST-09 の `definitions.max_positions` など）。"""
        v = (self.raw.get("definitions") or {}).get("max_positions")
        return int(v) if v is not None else None

    @property
    def params(self) -> dict[str, Any]:
        if self.param_values is not None:
            return dict(self.param_values)
        return {p["name"]: p["default"] for p in self.raw.get("params_to_optimize", [])}

    @property
    def param_specs(self) -> list[dict[str, Any]]:
        return list(self.raw.get("params_to_optimize", []))

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.raw.get("params_to_optimize", []))


_REQUIRED = ("id", "name", "factor", "hypothesis", "entry", "exit",
             "params_to_optimize", "deviations", "expected_correlation", "data_required")


def load_common(catalog_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((catalog_dir / "common.yaml").read_text(encoding="utf-8"))


def load_spec(path: Path, common: dict[str, Any]) -> StrategySpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise SpecError(f"{path.name}: 必須キーがありません: {', '.join(missing)}")

    # DEC-004: 損切り価格のないシグナルは無効。仕様の時点で弾く。
    stop = raw.get("exit", {}).get("stop", {})
    if not stop.get("initial"):
        raise SpecError(f"{path.name}: exit.stop.initial が必要です（DEC-004 / RM-001）")

    if raw["id"] != path.stem:
        raise SpecError(f"{path.name}: id({raw['id']}) がファイル名と一致しません")

    defaults = {p["name"]: p["default"] for p in raw.get("params_to_optimize", [])}
    try:
        filled = substitute_params(copy.deepcopy(raw), defaults)
    except SpecError as e:
        raise SpecError(f"{path.name}: {e}") from e
    spec = StrategySpec(
        id=raw["id"], name=raw["name"], factor=raw["factor"],
        phase=raw.get("phase"), common=common, raw=filled, template=raw, param_values=defaults,
    )
    _ = spec.rank          # 説明文のままの rank は起動時に弾く（実行中に辞書順採用へ落ちないように）
    return spec


def load_all(catalog_dir: Path, *, phase: int | None = None) -> list[StrategySpec]:
    common = load_common(catalog_dir)
    specs = [load_spec(p, common) for p in sorted((catalog_dir / "rules").glob("*.yaml"))]
    if phase is not None:
        specs = [s for s in specs if s.phase == phase]
    return specs


def assert_selected(catalog_dir: Path, spec_ids: list[str]) -> None:
    """FR-C07: status が selected_* でない手法は合議に組み込めない。"""
    index = yaml.safe_load((catalog_dir / "strategies.yaml").read_text(encoding="utf-8"))
    status = {e["id"]: e["status"] for e in index["entries"]}
    bad = [i for i in spec_ids if not str(status.get(i, "")).startswith("selected")]
    if bad:
        raise SpecError(
            f"代表手法に選抜されていない戦略は稼働できません（FR-C07）: {', '.join(bad)}"
        )
