docker build -t yt8m:gpu --build-arg GITHUB_TOKEN=$GITHUB_TOKEN -f pipeline/Dockerfile.gpu ./pipeline
