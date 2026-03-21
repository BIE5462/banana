from __future__ import annotations

from typing import Any, Callable

from license_service import LicenseManager
from license_store import load_license_options


def authorize_before_launch(
    *,
    dialog_factory: Callable[[LicenseManager], Any],
    show_error: Callable[[str], None],
    accepted_code: int,
) -> tuple[int | None, LicenseManager | None]:
    success, message, license_options = load_license_options()
    if not success:
        show_error(message)
        return 1, None

    license_manager = LicenseManager(license_options)
    if not license_manager.is_enabled:
        return None, license_manager

    dialog = dialog_factory(license_manager)
    if dialog.exec() != accepted_code:
        return 0, license_manager

    return None, license_manager
