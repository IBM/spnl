class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.22.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.22.0/spnl-v0.22.0-macos-aarch64.tar.gz"
      sha256 "92eca9e10ad2ba0923245b954d87e1c64cef369004fd1fce6f6bf15fad224a9f"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.22.0/spnl-v0.22.0-linux-aarch64-gnu.tar.gz"
      sha256 "1ac278ded5bdbb7539771b8b7fdf41042bc1d0141bacabb58624fe05580d7c86"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.22.0/spnl-v0.22.0-linux-x86_64-gnu.tar.gz"
      sha256 "8b879cf875d4e3aa89d743b15a82b04d3eb8418662e488f94d746da355f82234"
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
