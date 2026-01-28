class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.13.7"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v#{version}/spnl-v#{version}-macos-aarch64.tar.gz"
      sha256 "06946cfddaff92fea4d0c729bd562abf212c465202984c21a247644309f4258f"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v#{version}/spnl-v#{version}-linux-aarch64-gnu.tar.gz"
      sha256 "81fbb8e0c0b7a1e7394347193ec79f32b9de28cdec5c7f5065726876fc56cc9d"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v#{version}/spnl-v#{version}-linux-x86_64-gnu.tar.gz"
      sha256 "fe466f52020171ce41dfc00eece947b997aecce4ef62f2d56b5c81b193d937bd"
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
