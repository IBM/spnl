class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.14.3"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.14.3/spnl-v0.14.3-macos-aarch64.tar.gz"
      sha256 "98e9618a6e605159f980ecd8d87a9868484615383c91458cd10d0697feda8069"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.14.3/spnl-v0.14.3-linux-aarch64-gnu.tar.gz"
      sha256 "972aeaa2c4cb95c2f9ccd3496ce2396ffbfa7a748aea750cd49575124fc5ab69"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.14.3/spnl-v0.14.3-linux-x86_64-gnu.tar.gz"
      sha256 "89e03ba5a4567aea6f9244564ac518c1c89835a9b63f810b731f88f778a3c4c8"
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
