"""
A strict stand-in for the parts of Streamlit that app.py uses, so the app script can be executed and
tested where Streamlit is not installed (and in CI).

It is deliberately STRICT, to catch the mistakes that actually break Streamlit apps:
  * set_page_config must be the first Streamlit call;
  * slider / number_input arguments must not mix int and float, and value must lie in [min, max];
  * two widgets with the same type and label in one run are rejected (Streamlit's DuplicateWidgetID);
  * unknown attribute names (typos such as st.metrics) raise AttributeError;
  * metric / dataframe / pyplot / download_button receive the argument types Streamlit accepts.

It cannot prove the app works in the real Streamlit (version differences, browser behaviour);
it proves the script runs end to end and uses the API consistently.
"""

import os
import runpy
import sys

ALLOWED_API = {
    "set_page_config", "title", "header", "subheader", "caption", "markdown", "write", "text", "code", "info",
    "warning", "error", "success", "metric", "dataframe", "table", "pyplot", "columns", "tabs", "expander",
    "sidebar", "selectbox", "slider", "number_input", "checkbox", "radio", "button", "download_button",
    "spinner", "divider", "json", "stop",
}


class StopApp(Exception):
    """Raised by st.stop()."""


class Container:
    """What tabs / columns / expanders / the sidebar return: usable with `with`, forwards the API."""

    def __init__(self, stub):
        self._stub = stub

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        if name.startswith("_") or name not in ALLOWED_API:
            raise AttributeError(f"Streamlit container has no attribute {name!r}")
        return getattr(self._stub, name)


def _numeric_check(label, min_value, max_value, value, step):
    vals = [v for v in (min_value, max_value, value, step) if v is not None]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        raise TypeError(f"{label}: numeric arguments expected")
    kinds = {type(v) for v in vals}
    if len(kinds) > 1:
        raise TypeError(f"{label}: mixed numeric types {sorted(k.__name__ for k in kinds)} "
                        f"(Streamlit raises StreamlitMixedNumericTypesError)")
    if min_value is not None and max_value is not None and value is not None and not (min_value <= value <= max_value):
        raise ValueError(f"{label}: value {value} outside [{min_value}, {max_value}]")


class StubStreamlit:
    def __init__(self, overrides=None):
        self.overrides = dict(overrides or {})
        self.calls = []                    # (name, label-or-first-arg)
        self._keys = set()
        self._started = False
        self.sidebar = Container(self)

    # ---- bookkeeping ---------------------------------------------------------
    def _rec(self, name, first=None):
        if not self._started and name != "set_page_config":
            raise AssertionError(f"st.{name}() called before st.set_page_config(): it must be the first Streamlit call")
        self._started = True
        self.calls.append((name, first))

    def _widget(self, kind, label):
        key = (kind, label)
        if key in self._keys:
            raise RuntimeError(f"DuplicateWidgetID: two {kind} widgets with the label {label!r}")
        self._keys.add(key)

    def names(self, name):
        return [c for c in self.calls if c[0] == name]

    def __getattr__(self, name):
        raise AttributeError(f"module 'streamlit' has no attribute {name!r}")

    # ---- page and text -------------------------------------------------------------
    def set_page_config(self, page_title=None, page_icon=None, layout="centered", **kw):
        if self._started:
            raise AssertionError("st.set_page_config() must be the first Streamlit command")
        assert layout in ("centered", "wide")
        self._rec("set_page_config", page_title)

    def _text(name):
        def f(self, body="", *a, **kw):
            assert isinstance(body, str), f"st.{name} expects a string, got {type(body).__name__}"
            self._rec(name, body)
        return f

    title = _text("title"); header = _text("header"); subheader = _text("subheader"); caption = _text("caption")
    markdown = _text("markdown"); text = _text("text"); code = _text("code"); info = _text("info")
    warning = _text("warning"); error = _text("error"); success = _text("success")

    def write(self, *args, **kw):
        self._rec("write", args[0] if args else None)

    def divider(self):
        self._rec("divider")

    def json(self, body, **kw):
        self._rec("json")

    def stop(self):
        self._rec("stop")
        raise StopApp()

    # ---- layout --------------------------------------------------------------------------
    def tabs(self, labels):
        assert labels and all(isinstance(l, str) for l in labels), "st.tabs needs a non-empty list of strings"
        self._rec("tabs", tuple(labels))
        return [Container(self) for _ in labels]

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        assert n >= 1
        self._rec("columns", n)
        return [Container(self) for _ in range(n)]

    def expander(self, label, expanded=False, **kw):
        assert isinstance(label, str)
        self._rec("expander", label)
        return Container(self)

    def spinner(self, text="", **kw):
        self._rec("spinner", text)
        return Container(self)

    # ---- widgets ---------------------------------------------------------------------------
    def selectbox(self, label, options, index=0, **kw):
        options = list(options)
        assert options, f"{label}: selectbox needs options"
        self._widget("selectbox", label)
        self._rec("selectbox", label)
        out = self.overrides.get(label, options[index])
        assert out in options, f"{label}: {out!r} is not one of the options"
        return out

    def radio(self, label, options, index=0, horizontal=False, **kw):
        options = list(options)
        assert options
        self._widget("radio", label)
        self._rec("radio", label)
        out = self.overrides.get(label, options[index])
        assert out in options, f"{label}: {out!r} is not one of the options"
        return out

    def checkbox(self, label, value=False, **kw):
        assert isinstance(value, bool)
        self._widget("checkbox", label)
        self._rec("checkbox", label)
        return self.overrides.get(label, value)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        _numeric_check(label, min_value, max_value, value, step)
        self._widget("slider", label)
        self._rec("slider", label)
        return self.overrides.get(label, value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        _numeric_check(label, min_value, max_value, value, step)
        self._widget("number_input", label)
        self._rec("number_input", label)
        return self.overrides.get(label, value)

    def button(self, label, **kw):
        self._widget("button", label)
        self._rec("button", label)
        return False

    def download_button(self, label, data, file_name=None, mime=None, **kw):
        assert isinstance(data, (str, bytes)), f"download_button data must be str or bytes, got {type(data).__name__}"
        assert file_name is None or isinstance(file_name, str)
        self._widget("download_button", label)
        self._rec("download_button", label)
        return False

    # ---- data display -------------------------------------------------------------------------
    def metric(self, label, value, delta=None, delta_color="normal", help=None, **kw):
        assert isinstance(label, str)
        assert isinstance(value, (str, int, float)), f"metric value must be str/number, got {type(value).__name__}"
        assert delta is None or isinstance(delta, (str, int, float))
        assert delta_color in ("normal", "inverse", "off")
        self._rec("metric", label)

    def dataframe(self, data, hide_index=None, **kw):
        assert hasattr(data, "to_dict") and hasattr(data, "columns"), "st.dataframe needs a DataFrame here"
        self._rec("dataframe", len(data))

    def table(self, data, **kw):
        self._rec("table")

    def pyplot(self, fig=None, **kw):
        assert fig is not None and hasattr(fig, "savefig"), "st.pyplot needs a matplotlib Figure"
        self._rec("pyplot")


def run_app(app_path, overrides=None):
    """Execute the app script once with the stand-in installed as `streamlit`; returns the stub."""
    stub = StubStreamlit(overrides)
    sys.modules["streamlit"] = stub
    try:
        try:
            runpy.run_path(app_path, run_name="__main__")
        except StopApp:
            pass
    finally:
        sys.modules.pop("streamlit", None)
    return stub
