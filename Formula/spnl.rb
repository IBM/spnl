class Spnl < Formula
  desc "Span Query library for optimizing LLM inference costs"
  homepage "https://github.com/IBM/spnl"
  version "0.20.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.20.0/spnl-v0.20.0-macos-aarch64.tar.gz"
      sha256 "1eab903ecb263b9e40e89c32de2cb02a64b4f3274dbc0ebe43639a41fdf91841"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/IBM/spnl/releases/download/v0.20.0/spnl-v0.20.0-linux-aarch64-gnu.tar.gz"
      sha256 "066ddb733a64ef71f8f9edc72c221b0f10715642e8d033d5a00e07241823f2ef"
    end
    on_intel do
      url "https://github.com/IBM/spnl/releases/download/v0.20.0/spnl-v0.20.0-linux-x86_64-gnu.tar.gz"
      sha256 "04c065739b0c642d40e5ac7cb437cfb94b98ecd118e28b44661c5a291d8a9dff"
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
