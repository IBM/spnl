class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.17.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.17.0/spnl-v0.17.0-macos-aarch64.tar.gz"
      sha256 "8da4a1fd67904640bbc43de570127f5d78507a5d3a33a43dcac3a4580eac491b"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.17.0/spnl-v0.17.0-linux-aarch64-gnu.tar.gz"
      sha256 "f5183c27959103365131b1b29a5488489f3e34a0537a4268665f4db77f10f027"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.17.0/spnl-v0.17.0-linux-x86_64-gnu.tar.gz"
      sha256 "9c23b1739059bed0aa7010e2024db3ac4d18900f5150186469beadc586c04be5"
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
