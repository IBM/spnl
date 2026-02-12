class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.16.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.16.0/spnl-v0.16.0-macos-aarch64.tar.gz"
      sha256 "b0169f9ca052f4f4482ec8c443dbe8b1b6840104312456886b444d7a9a0979a0"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.16.0/spnl-v0.16.0-linux-aarch64-gnu.tar.gz"
      sha256 "79b9498e57c8b97a7f3accca576e904476f7e1d8c467b4be0170d5f83f8d1174"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.16.0/spnl-v0.16.0-linux-x86_64-gnu.tar.gz"
      sha256 "31520145b791dd5a57b34037b7470ee3fb15de3de9ed4be41afac7ecd8b5627e"
    end
  end

  livecheck do
    url :stable
    strategy :github_latest
  end

  def install
    bin.install "spnl"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/spnl --version")
  end
end

# Made with Bob
