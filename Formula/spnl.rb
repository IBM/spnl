class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.19.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/archive/refs/tags/v0.20.1.tar.gz"
      sha256 "c70cfc98d8a2eca0b470725e9a74be86a7ad5c55159cafe3efde8daf949c7c73"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.19.0/spnl-v0.19.0-linux-aarch64-gnu.tar.gz"
      sha256 "4a175ed5ee78dadbb7fb88aee633673a6ccc5b8fc9b1e3f3c402f8fe621ffa3a"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.19.0/spnl-v0.19.0-linux-x86_64-gnu.tar.gz"
      sha256 "5253cc4320143c34ed33a58c2a428ed203f9c4c4fd67f87e6b394813df5dc8d1"
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
