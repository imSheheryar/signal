from signal_earnings.text import lexical_features


def test_empty_text_returns_zeros():
    features = lexical_features("   ")
    assert features["n_words"] == 0
    assert features["hedge_rate"] == 0.0
    assert features["ttr"] == 0.0


def test_counts_and_rates_are_consistent():
    text = "We may see weakness next quarter. Revenue could decline 5%."
    features = lexical_features(text)

    assert features["n_words"] == 9
    assert features["n_numbers"] == 1
    assert features["hedge_rate"] == 100 * 2 / 9  # may, could
    assert features["neg_rate"] == 100 * 2 / 9  # weakness, decline
    assert features["fwd_rate"] == 100 * 1 / 9  # next
    assert features["modal_rate"] == 100 * 2 / 9  # may, could


def test_repeated_words_lower_type_token_ratio():
    assert lexical_features("risk")["ttr"] == 1.0
    assert lexical_features("risk risk risk")["ttr"] == 1 / 3
