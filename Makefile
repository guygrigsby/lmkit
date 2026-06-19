# lmkit Go ops CLI build targets. (The Python training library uses pyproject/pip.)
BIN       := lmkit
CMD       := ./cmd/lmkit
PLATFORMS := linux/amd64 linux/arm64 darwin/amd64 darwin/arm64

.PHONY: build test vet fmt install-cli dist clean

build:
	go build $(CMD)

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -l cmd internal

# install-cli compiles and installs the lmkit binary to $(go env GOBIN) (or
# $(go env GOPATH)/bin). Put that dir on PATH to run `lmkit` anywhere.
install-cli:
	go install $(CMD)

# dist cross-compiles static binaries for every release platform into dist/.
# CGO is off (pure Go: ssh is shelled out, no cgo deps) so cross-compiles cleanly.
dist:
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; \
	  out=dist/$(BIN)-$$os-$$arch; \
	  echo "building $$out"; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -trimpath -o $$out $(CMD) || exit 1; \
	done

clean:
	rm -rf dist
