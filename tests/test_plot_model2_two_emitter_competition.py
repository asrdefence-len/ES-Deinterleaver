from plot_model2_two_emitter_competition import plot_competition


def test_plot_competition_writes_image(tmp_path):
    matrix_off = {
        "46": {"46": 1, "38": 0, "neither": 1},
        "38": {"46": 0, "38": 1, "neither": 0},
        "neither": {"46": 0, "38": 0, "neither": 10},
    }
    matrix_on = {
        "46": {"46": 2, "38": 0, "neither": 0},
        "38": {"46": 0, "38": 1, "neither": 0},
        "neither": {"46": 0, "38": 0, "neither": 10},
    }
    def summary(matrix):
        return {
            "categories": ["46", "38", "neither"],
            "matrix_truth_rows_prediction_columns": matrix,
            "per_track": {
                "46": {"true_positive": matrix["46"]["46"]},
                "38": {"true_positive": matrix["38"]["38"]},
            },
            "cross_assignments": 0,
            "background_false_accepts": 0,
        }
    result = {
        "input_boundary": {
            "track_ids": ["46", "38"],
            "held_out_start_toa_s": 1.0,
            "window_pdws": 13,
        },
        "full_window_results": {
            "timing_off": summary(matrix_off),
            "timing_on": summary(matrix_on),
        },
        "timing_comparison": {"target_rescued": 1},
    }
    records = [
        {
            "toa_s": 1.1,
            "evaluation_category": "46",
            "timing_off": {
                "selected_track": None,
                "tracks": {"46": {"model2_margin": -0.2}},
            },
            "timing_on": {"selected_track": "46"},
        },
        {
            "toa_s": 1.2,
            "evaluation_category": "46",
            "timing_off": {
                "selected_track": "46",
                "tracks": {"46": {"model2_margin": 1.0}},
            },
            "timing_on": {"selected_track": "46"},
        },
        {
            "toa_s": 1.3,
            "evaluation_category": "38",
            "timing_off": {
                "selected_track": "38",
                "tracks": {"38": {"model2_margin": 0.8}},
            },
            "timing_on": {"selected_track": "38"},
        },
    ]
    output = tmp_path / "competition.png"
    plot_competition(result, records, output)
    assert output.exists()
    assert output.stat().st_size > 1000
