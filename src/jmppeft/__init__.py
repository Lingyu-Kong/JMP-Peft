import os

if not bool(int(os.environ.get("JMPPEFT_NO_JAXTYPING", "0"))):
    from ll.typecheck import typecheck_this_module

    typecheck_this_module()
