# Release a new version

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pypi_token>"
    exit 1
fi
PYPI_TOKEN="$1"
shift

if [[ $(git status --porcelain) ]]; then
  echo "There are uncommitted changes. Please commit or stash them before proceeding."
  exit 1
fi

# Bump the version and tag the release
PACKAGE_VERSION=$(uv version --bump patch --short)
git add .
git commit -m "v$PACKAGE_VERSION"
git tag "v$PACKAGE_VERSION"

# Publish to PyPI
rm -rf dist
uv build
uv publish --token "$PYPI_TOKEN" "$@"
