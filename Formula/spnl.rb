class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.21.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/archive/refs/tags/v0.22.1.tar.gz"
      sha256 "9ffcc814fdb41d1205354cfc8459240b89346a653d4b0bd9626526bb32b94eb9"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.21.0/spnl-v0.21.0-linux-aarch64-gnu.tar.gz"
      sha256 "784bd7996e57e87d6b0e3197f8eed057fac634490aac6deca482acfd76263d47"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.21.0/spnl-v0.21.0-linux-x86_64-gnu.tar.gz"
      sha256 "d4e664f258ed383e018e69a0ef9dc3c7032eeab436a499584590de357d2d8c18"
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
