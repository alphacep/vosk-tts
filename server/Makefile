GRPC_SOURCES = tts_service_pb2.py tts_service_pb2_grpc.py

all: $(GRPC_SOURCES)

$(GRPC_SOURCES): tts_service.proto
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tts_service.proto

clean:
	rm $(GRPC_SOURCES)
