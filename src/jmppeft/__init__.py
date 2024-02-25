import os

if not bool(int(os.environ.get("JMPPEFT_NO_JAXTYPING", "0"))):

    def _inner():
        from jaxtyping import install_import_hook

        install_import_hook(["jmppeft"], "beartype.beartype")

    _inner()
    del _inner
