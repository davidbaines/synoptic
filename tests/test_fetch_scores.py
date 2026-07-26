import pytest

from synoptic.fetch_scores import scores_from_log

LOG = """\
Generating hne/GEN: 250 verses ...
  chrF3=53.41 (copy=23.64, other=23.64 [hin]) truncated=0
METRICS_CSV_BEGIN
translation,language,book,verses,chrF3,copy_chrF3
hne,hne,GEN,250,53.41,23.64
hne,hne,[epistles],274,44.93,20.63
METRICS_CSV_END
  uploaded run archive as 10 parts + manifest to ClearML task abc
"""


def test_parses_csv_block():
    table = scores_from_log(LOG)
    assert len(table) == 2
    assert table.iloc[0]["chrF3"] == 53.41
    assert table.iloc[1]["book"] == "[epistles]"


def test_missing_block_raises():
    with pytest.raises(ValueError, match="METRICS_CSV"):
        scores_from_log("no markers here")
