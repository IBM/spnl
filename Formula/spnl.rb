class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.16.1"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.16.1/spnl-v0.16.1-macos-aarch64.tar.gz"
      sha256 "96222fb2b5a1bcde824260119ac791916a8857c9f58c0fc616f5f1aedf622613"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.16.1/spnl-v0.16.1-linux-aarch64-gnu.tar.gz"
      sha256 "22a3b7d74dc9a2ac74f50598c792bb82e276ffa41fbc6b0e54e3e71b590a2212"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.16.1/spnl-v0.16.1-linux-x86_64-gnu.tar.gz"
      sha256 "32ca8902b9ace8f62a84edd2174cd91241070677091d82106310b61aefc7ea27"
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
