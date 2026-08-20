#pragma once

#include <mpi.h>
#include <hdf5.h>
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>
#include <multi/array.hpp>
#include <multi/array_ref.hpp>
#include <exprtk.hpp>
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif
#include <spdlog/spdlog.h>
#include <pugixml.hpp>
