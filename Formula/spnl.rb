class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.14.2"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.14.2/spnl-v0.14.2-macos-aarch64.tar.gz"
      sha256 "cb251952d4b8c363d07fdc77e6c5a714860d88a0400ba33565060958e42c9810"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.14.2/spnl-v0.14.2-linux-aarch64-gnu.tar.gz"
      sha256 "14dc122259b3d23c834cf89ca96ad69c9450b2f8bb2fae37353b722103354172"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.14.2/spnl-v0.14.2-linux-x86_64-gnu.tar.gz"
      sha256 "5c484482a7e3fd721b2e92b99e9bea48cce8b7991ca628b90964832e8392a140"
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
