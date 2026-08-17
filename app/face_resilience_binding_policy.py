from __future__ import annotations

"""Keep AutomaticPipelineRunner bound to the final resilient face installer.

`automatic.py` imports `install_pretrained_face_handlers` and preflight by value.
Install the final face-domain guard before importing the automatic runner so both
bindings include the per-source identity firewall without a second recognition pass.
The V2 binding also makes a verified reference-guided damage consensus authoritative
before later INPAINT target discovery can broaden the repair domain again. V4 then
adds a fail-closed MAIN bridge: a pixel-verified same-canvas donor may override its
own preflight rejection and may extend trust only through its already frozen SFace
component. Reference-only agreement never becomes identity authority. The final V4
hardening requires face-local same-canvas evidence and a real SFace final comparison.
"""

_INSTALLED = False


def install_face_resilience_binding_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from app.reference_guided_seed_authority_policy import install_reference_guided_seed_authority_policy
    from app.face_domain_guard_v2_policy import install_face_domain_guard_v2_policy
    from app.identity_anchor_v4_policy import install_identity_anchor_v4_policy
    from app.identity_anchor_v4_hardening import install_identity_anchor_v4_hardening

    install_reference_guided_seed_authority_policy()
    install_face_domain_guard_v2_policy()
    install_identity_anchor_v4_policy()
    install_identity_anchor_v4_hardening()

    import app.automatic as automatic
    import app.preflight as preflight
    import app.pretrained_face_handlers as handlers

    automatic.preprocess_and_select_front_base = preflight.preprocess_and_select_front_base
    automatic.install_pretrained_face_handlers = handlers.install_pretrained_face_handlers
    _INSTALLED = True
