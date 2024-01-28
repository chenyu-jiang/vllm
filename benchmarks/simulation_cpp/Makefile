CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
OS := $(shell uname)
ifeq ($(OS),Darwin)
	CXXFLAGS += -undefined dynamic_lookup
endif

CPPFLAGS += $(shell python3 -m pybind11 --includes)
LIBNAME = simulator
LIBEXT = $(shell python3-config --extension-suffix)


default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): $(wildcard *.cpp) $(wildcard *.h) $(wildcard schedule_strategies/*.cpp) $(wildcard schedule_strategies/*.h)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@