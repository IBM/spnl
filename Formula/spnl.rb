class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.18.1"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.18.1/spnl-v0.18.1-macos-aarch64.tar.gz"
      sha256 "80100a6e46d3c0f971f62ca0424fdc54a285148cf38af0657ffe82f0e9249b2f"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.18.1/spnl-v0.18.1-linux-aarch64-gnu.tar.gz"
      sha256 "8bb0a2b86cd9d57ddeaedfadf336b67a7103f30fa90ff7842ab0f8237075d706"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.18.1/spnl-v0.18.1-linux-x86_64-gnu.tar.gz"
      sha256 "ddf2f2a794acf9685d4202d00fa81bbecac9424df646fba66227e66b96724e6f"
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
