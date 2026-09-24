from pathlib import Path

from paper.package_arxiv import build_package


def test_arxiv_package_is_self_contained(tmp_path):
    output = tmp_path / "arxiv"
    build_package(output)

    assert (output / "main.tex").exists()
    assert (output / "references.bib").exists()
    assert (output / "generated" / "primary_table.tex").exists()
    assert (output / "generated" / "rank_table.tex").exists()
    assert (output / "generated" / "mechanism_table.tex").exists()
    assert (output / "figures" / "architecture.png").exists()
    assert (output / "figures" / "final_loss_by_condition.png").exists()
    assert (output / "figures" / "parameter_efficiency.png").exists()
    assert r"\graphicspath{{figures/}}" in (output / "main.tex").read_text()
