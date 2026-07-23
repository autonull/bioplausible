from PyQt6.QtWidgets import QComboBox

from bioplausible.validation.tracks.track_registry import list_tracks


class TrackSelector(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("All Tracks", -1)
        tracks = list_tracks()
        for tid, name in tracks.items():
            # Cleanup name: track_42_something -> 42: Something
            display = f"{tid}: {name.replace('track_', '').replace(str(tid), '').replace('_', ' ').strip().title()}"
            self.addItem(display, tid)

    def get_selected_track_id(self):
        return self.currentData()
