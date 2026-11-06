from plot_model2_automatic_birth import plot_birth


def test_plot_birth_writes_image(tmp_path):
    def summary(accepted_46, rejected_46, accepted_38, rejected_38):
        return {
            "matrix_truth_rows_prediction_columns": {
                "46": {"46": accepted_46, "38": 0,
                       "rejected": rejected_46},
                "38": {"46": 0, "38": accepted_38,
                       "rejected": rejected_38},
            },
            "per_emitter": {
                "46": {"recall": accepted_46 / (accepted_46 + rejected_46)},
                "38": {"recall": accepted_38 / (accepted_38 + rejected_38)},
            },
            "tracks": [
                {
                    "track_id": "T1", "mapped_evaluation_emitter": "38",
                    "truth_counts_evaluation_only": {"38": accepted_38},
                    "rf_state_count": 2, "pulse_width_state_count": 1,
                },
                {
                    "track_id": "T2", "mapped_evaluation_emitter": "46",
                    "truth_counts_evaluation_only": {"46": accepted_46},
                    "rf_state_count": 3, "pulse_width_state_count": 2,
                },
            ],
        }
    result = {
        "input_boundary": {
            "selected_pdws": 7,
            "selected_truth_counts_evaluation_only": {"46": 4, "38": 3},
        },
        "timing_off": summary(3, 1, 2, 1),
        "timing_on": summary(4, 0, 3, 0),
        "timing_comparison": {"target_rescued": 2},
    }
    records = []
    for index, truth in enumerate(("38", "46", "38", "46", "38", "46", "46")):
        track_id = "T1" if truth == "38" else "T2"
        records.append({
            "toa_s": 1.0 + index * 1e-4,
            "evaluation_truth_label": int(truth),
            "timing_on": {
                "track_id": track_id,
                "outcome": "confirmed" if index in (4, 5)
                else "tentative_update",
            },
        })
    output = tmp_path / "birth.png"
    plot_birth(result, records, output)
    assert output.exists()
    assert output.stat().st_size > 1000
