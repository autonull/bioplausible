class SearchTabSchema:
    def get_layout(self):
        return {
            "type": "vbox",
            "children": [
                {
                    "type": "hbox",
                    "children": [
                        {"id": "task_selector", "type": "TaskSelector"},
                        {"id": "dataset_picker", "type": "DatasetPicker"},
                    ],
                },
                {"id": "model_selector", "type": "MultiModelSelector"},
                # New Tier Selector
                {
                    "type": "group",
                    "title": "Discovery Tier (Patience Level)",
                    "children": [
                        {
                            "type": "hbox",
                            "children": [
                                {"type": "label", "text": "Select Tier:"},
                                {
                                    "id": "tier_selector",
                                    "type": "combobox",
                                    "items": [
                                        "Smoke (1 min)",
                                        "Shallow (10 min)",
                                        "Standard (1 hr)",
                                        "Deep (Overnight)",
                                    ],
                                },
                                {
                                    "type": "label",
                                    "text": "  (Controls epochs, trials, and search depth)",
                                },
                            ],
                        }
                    ],
                },
                {"id": "results_table", "type": "ResultsTable"},
                {"id": "radar_view", "type": "RadarView"},
                {
                    "type": "hbox",
                    "children": [
                        {
                            "id": "start_btn",
                            "type": "button",
                            "text": "🔍 Start Search",
                            "action": "start",
                            "primary": True,
                        },
                        {
                            "id": "stop_btn",
                            "type": "button",
                            "text": "🛑 Stop",
                            "action": "stop",
                        },
                    ],
                },
                {"id": "log_output", "type": "log"},
            ],
        }


SEARCH_TAB_SCHEMA = SearchTabSchema().get_layout()
