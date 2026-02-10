class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.15.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.15.0/spnl-v0.15.0-macos-aarch64.tar.gz"
      sha256 "b3f61956826cca620171c919736b0d1fff9b853831575c31f6f0e6f20ab2ed31"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.15.0/spnl-v0.15.0-linux-aarch64-gnu.tar.gz"
      sha256 "45614c6102b77b231f31201f1a52353a9bbea18b825c85718073289e1cad90a1"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.15.0/spnl-v0.15.0-linux-x86_64-gnu.tar.gz"
      sha256 "cb803c877cde73ce09f60d29a3c6cc41b93d7570627c9e3d6406d52a501fc76d"
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
