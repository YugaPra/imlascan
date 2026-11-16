import os
import threading

from django.apps import AppConfig


class HomeConfig(AppConfig):
    name = 'home'

    def ready(self):
        # IMLASCAN_PRELOAD=True (disarankan di server): muat model AraT5 saat server menyala,
        # di latar, supaya pengunjung pertama tidak menunggu pemuatan model.
        if os.environ.get("IMLASCAN_PRELOAD", "").strip().lower() in ("1", "true", "yes"):
            from . import gec_model
            threading.Thread(target=gec_model._lazy_load, name="imlascan-preload", daemon=True).start()
