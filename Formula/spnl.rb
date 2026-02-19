class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.20.1"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.20.1/spnl-v0.20.1-macos-aarch64.tar.gz"
      sha256 "733d6037b17a24b689fa21479fbe16f752671646cee250c651fdd8bacff3105d"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.20.1/spnl-v0.20.1-linux-aarch64-gnu.tar.gz"
      sha256 "c741ba45f416619f67321db2470b15353ecd234c1ee13f070448cc8345c8a57e"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.20.1/spnl-v0.20.1-linux-x86_64-gnu.tar.gz"
      sha256 "38a97bb9e8eafee6cdc913bcf1e50ade6f4a81bf2514b2a971396bf195436507"
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
