from ressim.cli import main


def test_buckley_leverett_command(capsys):
    main(["buckley-leverett", "--pvi", "1.0"])
    out = capsys.readouterr().out
    assert "Breakthrough" in out and "0.4452" in out


def test_waterflood_command(capsys, tmp_path):
    main(["waterflood", "--nx", "40", "--pvi", "0.5", "--profiles", "0.2", "--plot", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Mass balance error" in out
    assert (tmp_path / "profiles.png").exists()


def test_five_spot_command(capsys):
    main(["five-spot", "--n", "8", "--size", "80", "--pvi", "0.3", "--solver", "gmres"])
    assert "Breakthrough" in capsys.readouterr().out


def test_validate_quick(capsys, tmp_path):
    main(["validate", "--quick", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "observed order" in out
    assert (tmp_path / "bl_profiles.png").exists()
