class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.22.1"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.22.1/spnl-v0.22.1-macos-aarch64.tar.gz"
      sha256 "120b4605f6ad585636c346df1f58d4d571cc81d410b9abd0132d7fdd540e8192"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.22.1/spnl-v0.22.1-linux-aarch64-gnu.tar.gz"
      sha256 "f0a8c5acfd25375dab074cc205a9381ecd136f4f28a16e1c90cffa85f464d7f8"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.22.1/spnl-v0.22.1-linux-x86_64-gnu.tar.gz"
      sha256 "5cf8c1d03099ad74141b9728ffa3b4bbd5898a244454866861e812f26ebf36ff"
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
