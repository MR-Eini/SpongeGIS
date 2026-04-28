# ============================================================
# Option B: Rule-based vector-river valley / floodplain model
# ============================================================
# Purpose
#   Replace the weak DEM-only class 2 = valley/floodplain rule
#   with a simpler rule-based method anchored to mapped vector rivers.
#
# Main idea
#   Class 2 is identified from:
#     - distance to vector river
#     - low slope
#     - low/neutral topographic position index (TPI)
#     - optional alluvial-soil/floodplain support layer
#
# Final classes:
#   1 = vector-river corridor / channel proxy
#   2 = river valley / floodplain proxy
#   3 = upland / remaining terrain
#
# Notes
#   - This method does NOT require manual floodplain polygons.
#   - It is not formally calibrated. Use the sensitivity output to
#     choose defensible thresholds.
#   - Because vector rivers are used as the spatial anchor, this
#     method is less affected by wrong DTM-derived drainage paths.
#   - DEM smoothing is included to reduce small artificial embankment
#     effects in slope and TPI predictors.
# ============================================================

# -----------------------------
# 0) PACKAGES
# -----------------------------
required_packages <- c("terra")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Please install missing package(s): ", paste(missing_packages, collapse = ", "))
}

library(terra)

# -----------------------------
# 1) USER SETTINGS
# -----------------------------
# Main inputs
# These paths were set for the Kamienna case. Adjust if needed.
dem_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/DEM50.tif"
river_vector_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/River/River.shp"

# Optional supporting layer.
# Can be a raster or vector layer. Use NA_character_ if unavailable.
# Examples:
# alluvial_soil_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/alluvial_soils/alluvial_soils.shp"
# alluvial_soil_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/alluvial_soils.tif"
alluvial_soil_file <- NA_character_

# Output folder
out_dir <- "X:/download/Mamad/!SpongeGIStest/Kamienna/paper_zone/OptionB_vector_river_rule_based_valley"

# DEM smoothing for valley/floodplain terrain predictors.
# This is useful when embankments or small artificial structures create sharp local slopes.
smooth_dem_for_valley <- TRUE
smooth_window_cells <- 5  # must be odd; 3 or 5 is usually enough

# TPI neighborhood sizes in meters.
# Large TPI is used for valley/floodplain classification.
tpi_small_window_m <- 250
tpi_large_window_m <- 1000

# Class 1: narrow vector-river corridor / channel proxy.
channel_buffer_m <- 50

# Main Option B thresholds for class 2.
# These are starting values for a 50 m DEM. Inspect sensitivity outputs and maps.
valley_max_distance_m <- 500
valley_max_slope_deg <- 5
valley_tpi_large_threshold_m <- 1

# Optional additional local TPI constraint.
# If use_tpi_small_constraint = TRUE, class 2 must also satisfy tpi_small <= threshold.
use_tpi_small_constraint <- FALSE
valley_tpi_small_threshold_m <- 1

# Optional alluvial-soil mode:
#   "ignore"  = do not use alluvial soils
#   "require" = candidate valley cells must also be alluvial/floodplain cells
#   "expand"  = add nearby alluvial/floodplain cells to the terrain-derived candidate
# If alluvial_soil_file is NA, this is automatically changed to "ignore".
alluvial_mode <- "ignore"  # "ignore", "require", or "expand"
alluvial_expand_distance_m <- 1000
alluvial_expand_slope_deg <- 8

# Optional cleanup.
apply_majority_filter_to_valley <- FALSE
majority_window_size <- 3

# Optional upland split.
split_upland <- TRUE
upland_split_slope_deg <- 6

# Sensitivity analysis ranges.
# These do not affect the final map unless you manually copy one of the selected combinations above.
sensitivity_distance_m <- c(100, 200, 300, 500, 750, 1000)
sensitivity_slope_deg <- c(2, 3, 5, 8, 12)
sensitivity_tpi_large_m <- c(-2, -1, 0, 1, 2)

# Save selected sensitivity rasters for visual inspection.
save_sensitivity_rasters <- TRUE
sensitivity_raster_limit <- 20

# -----------------------------
# 2) HELPER FUNCTIONS
# -----------------------------
stop_if_missing <- function(path, label) {
  if (is.na(path) || !nzchar(path) || !file.exists(path)) {
    stop(label, " not found: ", path)
  }
}

make_odd <- function(x) {
  x <- as.integer(round(x))
  if (x %% 2 == 0) x <- x + 1
  if (x < 3) x <- 3
  x
}

extent_overlaps <- function(e1, e2) {
  # e1/e2 are SpatExtent objects. Index order: xmin, xmax, ymin, ymax.
  (e1[1] <= e2[2]) && (e1[2] >= e2[1]) && (e1[3] <= e2[4]) && (e1[4] >= e2[3])
}

safe_crs_text <- function(x) {
  y <- try(crs(x), silent = TRUE)
  if (inherits(y, "try-error") || is.na(y) || !nzchar(y)) return(NA_character_)
  y
}

inspect_vector_against_template <- function(file, template, layer_name, out_dir) {
  diag_file <- file.path(out_dir, paste0(layer_name, "_spatial_diagnostics.csv"))
  v <- try(vect(file), silent = TRUE)

  if (inherits(v, "try-error")) {
    d <- data.frame(
      layer_name = layer_name,
      file = normalizePath(file, winslash = "/", mustWork = FALSE),
      readable = FALSE,
      n_features = NA_integer_,
      geom_type = NA_character_,
      crs = NA_character_,
      xmin = NA_real_, xmax = NA_real_, ymin = NA_real_, ymax = NA_real_,
      projected_to_template = NA,
      overlaps_template = NA,
      stringsAsFactors = FALSE
    )
    write.csv(d, diag_file, row.names = FALSE)
    stop(layer_name, " could not be read: ", file)
  }

  d_original <- data.frame(
    layer_name = paste0(layer_name, "_original"),
    file = normalizePath(file, winslash = "/", mustWork = FALSE),
    readable = TRUE,
    n_features = nrow(v),
    geom_type = ifelse(nrow(v) > 0, paste(unique(geomtype(v)), collapse = ";"), NA_character_),
    crs = safe_crs_text(v),
    xmin = ifelse(nrow(v) > 0, ext(v)[1], NA_real_),
    xmax = ifelse(nrow(v) > 0, ext(v)[2], NA_real_),
    ymin = ifelse(nrow(v) > 0, ext(v)[3], NA_real_),
    ymax = ifelse(nrow(v) > 0, ext(v)[4], NA_real_),
    projected_to_template = FALSE,
    overlaps_template = NA,
    stringsAsFactors = FALSE
  )

  if (nrow(v) == 0) {
    write.csv(d_original, diag_file, row.names = FALSE)
    stop(layer_name, " vector has zero features: ", file)
  }

  if (is.na(crs(v)) || !nzchar(crs(v))) {
    write.csv(d_original, diag_file, row.names = FALSE)
    stop(layer_name, " has no CRS. Define its correct CRS before using it: ", file)
  }

  vp <- try(project(v, crs(template)), silent = TRUE)
  if (inherits(vp, "try-error")) {
    write.csv(d_original, diag_file, row.names = FALSE)
    stop(layer_name, " could not be projected to DEM CRS. Check CRS definition: ", file)
  }

  overlaps <- extent_overlaps(ext(vp), ext(template))

  d_projected <- data.frame(
    layer_name = paste0(layer_name, "_projected_to_DEM"),
    file = normalizePath(file, winslash = "/", mustWork = FALSE),
    readable = TRUE,
    n_features = nrow(vp),
    geom_type = paste(unique(geomtype(vp)), collapse = ";"),
    crs = safe_crs_text(vp),
    xmin = ext(vp)[1], xmax = ext(vp)[2], ymin = ext(vp)[3], ymax = ext(vp)[4],
    projected_to_template = TRUE,
    overlaps_template = overlaps,
    stringsAsFactors = FALSE
  )

  d_template <- data.frame(
    layer_name = "DEM_template",
    file = NA_character_,
    readable = TRUE,
    n_features = NA_integer_,
    geom_type = "raster",
    crs = safe_crs_text(template),
    xmin = ext(template)[1], xmax = ext(template)[2], ymin = ext(template)[3], ymax = ext(template)[4],
    projected_to_template = NA,
    overlaps_template = NA,
    stringsAsFactors = FALSE
  )

  write.csv(rbind(d_template, d_original, d_projected), diag_file, row.names = FALSE)

  if (!overlaps) {
    stop(layer_name, " vector extent does not overlap the DEM after CRS projection. Diagnostics written to: ", diag_file)
  }

  vp
}

focal_mean <- function(x, window_m, res_m, name) {
  nwin <- make_odd(window_m / res_m)
  w <- matrix(1, nrow = nwin, ncol = nwin)
  xm <- focal(x, w = w, fun = mean, na.policy = "omit", fillvalue = NA)
  names(xm) <- name
  xm
}

make_tpi <- function(dem_x, window_m, res_m, name) {
  dem_mean <- focal_mean(dem_x, window_m, res_m, paste0(name, "_mean"))
  out <- dem_x - dem_mean
  names(out) <- name
  out
}

load_binary_layer <- function(file, template, layer_name, required_overlap = FALSE) {
  if (is.na(file) || !nzchar(file) || !file.exists(file)) {
    return(NULL)
  }

  extn <- tolower(tools::file_ext(file))

  if (extn %in% c("tif", "tiff", "img", "asc", "grd")) {
    r <- rast(file)
    if (is.na(crs(r)) || !nzchar(crs(r))) {
      stop(layer_name, " raster has no CRS: ", file)
    }
    if (!same.crs(r, template)) {
      r <- project(r, template, method = "near")
    }
    r <- resample(r, template, method = "near")
    r <- crop(r, template)
    r <- mask(r, template)
    out <- ifel(!is.na(r) & r != 0, 1, NA)
    names(out) <- layer_name
    return(out)
  }

  v <- inspect_vector_against_template(file, template, layer_name, out_dir)

  # Crop to DEM extent to avoid unnecessary rasterization work.
  v_crop <- try(crop(v, ext(template)), silent = TRUE)
  if (!inherits(v_crop, "try-error") && nrow(v_crop) > 0) {
    v <- v_crop
  }

  v[[layer_name]] <- 1
  out <- rasterize(v, template, field = layer_name, background = NA, touches = TRUE)
  out <- ifel(!is.na(out), 1, NA)
  names(out) <- layer_name

  n_pos <- as.numeric(global(!is.na(out), "sum", na.rm = TRUE)[1, 1])
  if (is.na(n_pos)) n_pos <- 0

  if (required_overlap && n_pos == 0) {
    stop(layer_name, " rasterized to zero cells although vector extent overlaps DEM. Check geometry validity and resolution.")
  }

  out
}

raster_cell_count <- function(x) {
  val <- as.numeric(global(x, "sum", na.rm = TRUE)[1, 1])
  if (is.na(val)) 0 else val
}

make_zone_summary <- function(zones, slope_deg, slope_pct, dem, tpi_large, dist_to_river) {
  # Robust summary function for older/newer terra versions.
  # Some terra versions do not support freq(..., useNA = "no"), so NA values are removed manually.

  freq_raw <- try(as.data.frame(freq(zones)), silent = TRUE)

  if (inherits(freq_raw, "try-error") || nrow(freq_raw) == 0) {
    stop("No valid zone cells were found in zones raster. Check classification outputs.")
  }

  nms <- names(freq_raw)

  if ("value" %in% nms && "count" %in% nms) {
    zone_col <- "value"
    count_col <- "count"
  } else {
    if (ncol(freq_raw) < 2) {
      stop("Unexpected terra::freq() output. Columns were: ", paste(nms, collapse = ", "))
    }
    zone_col <- nms[ncol(freq_raw) - 1]
    count_col <- nms[ncol(freq_raw)]
  }

  zone_freq <- data.frame(
    zone = suppressWarnings(as.integer(freq_raw[[zone_col]])),
    n_cells = suppressWarnings(as.numeric(freq_raw[[count_col]])),
    stringsAsFactors = FALSE
  )

  zone_freq <- zone_freq[!is.na(zone_freq$zone) & !is.na(zone_freq$n_cells), , drop = FALSE]
  zone_freq <- zone_freq[zone_freq$zone %in% c(1L, 2L, 3L), , drop = FALSE]

  if (nrow(zone_freq) == 0) {
    stop("The zones raster contains no valid classes 1, 2, or 3. Check the classification result.")
  }

  cell_area_m2 <- prod(res(zones))
  zone_freq$area_m2 <- zone_freq$n_cells * cell_area_m2
  zone_freq$area_km2 <- zone_freq$area_m2 / 1e6
  zone_freq$area_percent <- 100 * zone_freq$area_m2 / sum(zone_freq$area_m2, na.rm = TRUE)

  zone_names <- c(
    `1` = "channel_vector_corridor",
    `2` = "river_valley_floodplain_proxy",
    `3` = "upland"
  )
  zone_freq$zone_name <- unname(zone_names[as.character(zone_freq$zone)])
  zone_freq$zone_name[is.na(zone_freq$zone_name)] <- paste0("zone_", zone_freq$zone[is.na(zone_freq$zone_name)])

  zonal_one <- function(x, z, fun_name, out_name) {
    zz <- try(as.data.frame(zonal(x, z, fun = fun_name, na.rm = TRUE)), silent = TRUE)

    if (inherits(zz, "try-error") || nrow(zz) == 0 || ncol(zz) < 2) {
      return(data.frame(zone = integer(0), stringsAsFactors = FALSE))
    }

    out <- data.frame(
      zone = suppressWarnings(as.integer(zz[[1]])),
      stat_value = suppressWarnings(as.numeric(zz[[ncol(zz)]])),
      stringsAsFactors = FALSE
    )

    out <- out[!is.na(out$zone), , drop = FALSE]
    names(out)[2] <- out_name
    out
  }

  zone_stats <- data.frame(zone = sort(unique(zone_freq$zone)), stringsAsFactors = FALSE)

  stat_tables <- list(
    zonal_one(slope_deg,     zones, "mean", "slope_deg_mean"),
    zonal_one(slope_deg,     zones, "min",  "slope_deg_min"),
    zonal_one(slope_deg,     zones, "max",  "slope_deg_max"),
    zonal_one(slope_pct,     zones, "mean", "slope_pct_mean"),
    zonal_one(dem,           zones, "mean", "elevation_mean"),
    zonal_one(tpi_large,     zones, "mean", "tpi_large_mean"),
    zonal_one(dist_to_river, zones, "mean", "distance_to_river_mean_m")
  )

  for (tbl in stat_tables) {
    if (nrow(tbl) > 0) {
      zone_stats <- merge(zone_stats, tbl, by = "zone", all.x = TRUE)
    }
  }

  out <- merge(zone_freq, zone_stats, by = "zone", all = TRUE)
  out <- out[order(out$zone), , drop = FALSE]

  wanted <- c(
    "zone", "zone_name", "n_cells", "area_m2", "area_km2", "area_percent",
    "elevation_mean", "slope_deg_mean", "slope_deg_min", "slope_deg_max",
    "slope_pct_mean", "tpi_large_mean", "distance_to_river_mean_m"
  )

  missing_cols <- setdiff(wanted, names(out))
  if (length(missing_cols) > 0) {
    for (m in missing_cols) out[[m]] <- NA_real_
  }

  out[, wanted, drop = FALSE]
}
# -----------------------------
# 3) BASIC CHECKS AND DEM
# -----------------------------
stop_if_missing(dem_file, "DEM")
stop_if_missing(river_vector_file, "River vector")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

message("Reading DEM...")
dem <- rast(dem_file)

if (is.na(crs(dem)) || !nzchar(crs(dem))) {
  stop("The DEM has no CRS. A known projected CRS is needed.")
}

if (is.lonlat(dem)) {
  stop("The DEM is in longitude/latitude. Reproject it to a projected CRS in meters before running this workflow.")
}

res_xy <- res(dem)
if (abs(res_xy[1] - res_xy[2]) > 1e-9) {
  warning("Raster cells are not square. The script will run, but distance/window interpretation may be less consistent.")
}
res_m <- mean(res_xy)
valid <- !is.na(dem)

# -----------------------------
# 4) TERRAIN PREDICTORS
# -----------------------------
message("Preparing DEM-derived predictors...")

if (smooth_dem_for_valley) {
  smooth_window_cells <- make_odd(smooth_window_cells)
  w_smooth <- matrix(1, nrow = smooth_window_cells, ncol = smooth_window_cells)
  dem_valley <- focal(dem, w = w_smooth, fun = mean, na.policy = "omit", fillvalue = NA)
  names(dem_valley) <- "dem_smoothed_for_valley"
} else {
  dem_valley <- dem
  names(dem_valley) <- "dem_for_valley"
}

slope_deg <- terrain(dem_valley, v = "slope", unit = "degrees")
names(slope_deg) <- "slope_deg"

slope_pct <- tan(slope_deg * pi / 180) * 100
names(slope_pct) <- "slope_pct"

tpi_small <- make_tpi(dem_valley, tpi_small_window_m, res_m, "tpi_small")
tpi_large <- make_tpi(dem_valley, tpi_large_window_m, res_m, "tpi_large")

slope_class <- classify(
  slope_deg,
  rcl = rbind(
    c(-Inf,  2, 1),
    c(   2,  5, 2),
    c(   5, 10, 3),
    c(  10, 20, 4),
    c(  20, Inf, 5)
  ),
  include.lowest = TRUE,
  right = FALSE
)
names(slope_class) <- "slope_class"

# -----------------------------
# 5) VECTOR RIVER DISTANCE
# -----------------------------
message("Preparing vector-river distance layer...")
river_v <- inspect_vector_against_template(river_vector_file, dem, "river_vector", out_dir)
river_v_crop <- try(crop(river_v, ext(dem)), silent = TRUE)
if (!inherits(river_v_crop, "try-error") && nrow(river_v_crop) > 0) river_v <- river_v_crop

river_v[["river_id_for_raster"]] <- 1
river_cells <- rasterize(river_v, dem, field = "river_id_for_raster", background = NA, touches = TRUE)
river_cells <- ifel(!is.na(river_cells), 1, NA)
names(river_cells) <- "vector_river_cells"

river_cell_count <- raster_cell_count(!is.na(river_cells))
message("Rasterized river cells: ", river_cell_count)
if (river_cell_count == 0) {
  stop("Vector rivers rasterized to zero cells. Check CRS, extent, and DEM resolution.")
}

dist_to_river <- distance(river_cells)
names(dist_to_river) <- "distance_to_vector_river_m"

# -----------------------------
# 6) OPTIONAL ALLUVIAL / FLOODPLAIN SUPPORT LAYER
# -----------------------------
if (!is.na(alluvial_soil_file) && nzchar(alluvial_soil_file) && file.exists(alluvial_soil_file)) {
  message("Preparing optional alluvial/floodplain support layer...")
  alluvial_mask <- load_binary_layer(alluvial_soil_file, dem, "alluvial_or_floodplain_support", required_overlap = FALSE)
  if (is.null(alluvial_mask)) {
    alluvial_mode <- "ignore"
  }
} else {
  alluvial_mask <- NULL
  alluvial_mode <- "ignore"
}

if (!alluvial_mode %in% c("ignore", "require", "expand")) {
  stop("Invalid alluvial_mode. Use 'ignore', 'require', or 'expand'.")
}

message("Alluvial mode: ", alluvial_mode)

# -----------------------------
# 7) OPTION B CLASSIFICATION
# -----------------------------
message("Classifying Option B river-valley/floodplain proxy...")

channel <- valid & (dist_to_river <= channel_buffer_m)
names(channel) <- "class1_channel_vector_corridor"

valley_base <- valid &
  (dist_to_river <= valley_max_distance_m) &
  (slope_deg <= valley_max_slope_deg) &
  (tpi_large <= valley_tpi_large_threshold_m)

if (use_tpi_small_constraint) {
  valley_base <- valley_base & (tpi_small <= valley_tpi_small_threshold_m)
}

valley <- valley_base

if (!is.null(alluvial_mask) && alluvial_mode == "require") {
  valley <- valley & (!is.na(alluvial_mask))
}

if (!is.null(alluvial_mask) && alluvial_mode == "expand") {
  alluvial_expand <- valid &
    (!is.na(alluvial_mask)) &
    (dist_to_river <= alluvial_expand_distance_m) &
    (slope_deg <= alluvial_expand_slope_deg)
  valley <- valley | alluvial_expand
}

# Class 2 cannot overwrite class 1.
valley <- valid & valley & (!channel)
names(valley) <- "class2_rule_based_river_valley_floodplain"

if (apply_majority_filter_to_valley) {
  majority_window_size <- make_odd(majority_window_size)
  mw <- matrix(1, majority_window_size, majority_window_size)
  valley_num <- ifel(valley, 1, NA)
  valley_num <- focal(valley_num, w = mw, fun = modal, na.policy = "omit", fillvalue = NA)
  valley <- valid & (!is.na(valley_num)) & (!channel)
  names(valley) <- "class2_rule_based_river_valley_floodplain_filtered"
}

upland <- valid & (!channel) & (!valley)

zones_3 <- ifel(channel, 1, ifel(valley, 2, ifel(upland, 3, NA)))
names(zones_3) <- "zone"

if (split_upland) {
  upland_low  <- upland & (slope_deg <  upland_split_slope_deg)
  upland_high <- upland & (slope_deg >= upland_split_slope_deg)

  zones_4 <- ifel(channel, 1,
                  ifel(valley, 2,
                       ifel(upland_low, 3,
                            ifel(upland_high, 4, NA))))
  names(zones_4) <- "zone4"
}

# -----------------------------
# 8) SENSITIVITY ANALYSIS WITHOUT MANUAL POLYGONS
# -----------------------------
message("Running threshold sensitivity analysis...")

combo_grid <- expand.grid(
  distance_m = sensitivity_distance_m,
  slope_deg = sensitivity_slope_deg,
  tpi_large_m = sensitivity_tpi_large_m,
  stringsAsFactors = FALSE
)

cell_area_m2 <- prod(res(dem))
valid_cells <- raster_cell_count(valid)
valid_area_km2 <- valid_cells * cell_area_m2 / 1e6

sensitivity_rows <- vector("list", nrow(combo_grid))

sens_dir <- file.path(out_dir, "sensitivity_class2_rasters")
if (save_sensitivity_rasters) dir.create(sens_dir, showWarnings = FALSE, recursive = TRUE)

for (i in seq_len(nrow(combo_grid))) {
  d_thr <- combo_grid$distance_m[i]
  s_thr <- combo_grid$slope_deg[i]
  t_thr <- combo_grid$tpi_large_m[i]

  cand <- valid &
    (dist_to_river <= d_thr) &
    (slope_deg <= s_thr) &
    (tpi_large <= t_thr) &
    (!channel)

  if (use_tpi_small_constraint) {
    cand <- cand & (tpi_small <= valley_tpi_small_threshold_m)
  }

  if (!is.null(alluvial_mask) && alluvial_mode == "require") {
    cand <- cand & (!is.na(alluvial_mask))
  }

  if (!is.null(alluvial_mask) && alluvial_mode == "expand") {
    alluvial_expand <- valid &
      (!is.na(alluvial_mask)) &
      (dist_to_river <= alluvial_expand_distance_m) &
      (slope_deg <= alluvial_expand_slope_deg) &
      (!channel)
    cand <- cand | alluvial_expand
  }

  n_valley <- raster_cell_count(cand)
  area_km2 <- n_valley * cell_area_m2 / 1e6
  area_percent <- 100 * area_km2 / valid_area_km2

  sensitivity_rows[[i]] <- data.frame(
    distance_m = d_thr,
    slope_deg = s_thr,
    tpi_large_m = t_thr,
    n_valley_cells = n_valley,
    valley_area_km2 = area_km2,
    valley_area_percent_of_valid_dem = area_percent,
    stringsAsFactors = FALSE
  )

  if (save_sensitivity_rasters && i <= sensitivity_raster_limit) {
    cand_r <- ifel(cand, 1, NA)
    names(cand_r) <- "candidate_class2"
    out_name <- sprintf("candidate_class2_dist%04dm_slope%02d_tpi%+03d.tif", d_thr, s_thr, t_thr)
    out_name <- gsub("\\+", "p", out_name)
    out_name <- gsub("-", "m", out_name)
    writeRaster(cand_r, file.path(sens_dir, out_name), overwrite = TRUE)
  }
}

sensitivity_table <- do.call(rbind, sensitivity_rows)
sensitivity_table <- sensitivity_table[order(sensitivity_table$distance_m, sensitivity_table$slope_deg, sensitivity_table$tpi_large_m), ]

# -----------------------------
# 9) SUMMARIES AND METADATA
# -----------------------------
zone_summary <- make_zone_summary(zones_3, slope_deg, slope_pct, dem_valley, tpi_large, dist_to_river)

metadata <- data.frame(
  parameter = c(
    "dem_file", "river_vector_file", "alluvial_soil_file", "out_dir",
    "dem_resolution_m", "smooth_dem_for_valley", "smooth_window_cells",
    "tpi_small_window_m", "tpi_large_window_m", "channel_buffer_m",
    "valley_max_distance_m", "valley_max_slope_deg", "valley_tpi_large_threshold_m",
    "use_tpi_small_constraint", "valley_tpi_small_threshold_m",
    "alluvial_mode", "alluvial_expand_distance_m", "alluvial_expand_slope_deg",
    "split_upland", "upland_split_slope_deg", "apply_majority_filter_to_valley"
  ),
  value = as.character(c(
    dem_file, river_vector_file, alluvial_soil_file, out_dir,
    res_m, smooth_dem_for_valley, smooth_window_cells,
    tpi_small_window_m, tpi_large_window_m, channel_buffer_m,
    valley_max_distance_m, valley_max_slope_deg, valley_tpi_large_threshold_m,
    use_tpi_small_constraint, valley_tpi_small_threshold_m,
    alluvial_mode, alluvial_expand_distance_m, alluvial_expand_slope_deg,
    split_upland, upland_split_slope_deg, apply_majority_filter_to_valley
  )),
  stringsAsFactors = FALSE
)

# -----------------------------
# 10) SAVE OUTPUTS
# -----------------------------
message("Saving outputs...")

writeRaster(dem_valley,     file.path(out_dir, "dem_for_valley_predictors.tif"), overwrite = TRUE)
writeRaster(slope_deg,      file.path(out_dir, "slope_deg_smoothed_if_enabled.tif"), overwrite = TRUE)
writeRaster(slope_pct,      file.path(out_dir, "slope_pct_smoothed_if_enabled.tif"), overwrite = TRUE)
writeRaster(slope_class,    file.path(out_dir, "slope_class.tif"), overwrite = TRUE)
writeRaster(tpi_small,      file.path(out_dir, "tpi_small.tif"), overwrite = TRUE)
writeRaster(tpi_large,      file.path(out_dir, "tpi_large.tif"), overwrite = TRUE)
writeRaster(river_cells,    file.path(out_dir, "vector_river_cells.tif"), overwrite = TRUE)
writeRaster(dist_to_river,  file.path(out_dir, "distance_to_vector_river_m.tif"), overwrite = TRUE)
writeRaster(channel,        file.path(out_dir, "class1_vector_river_corridor.tif"), overwrite = TRUE)
writeRaster(valley,         file.path(out_dir, "class2_optionB_rule_based_valley_floodplain.tif"), overwrite = TRUE)
writeRaster(zones_3,        file.path(out_dir, "zones_3class_optionB_vectorRiver_valley_upland.tif"), overwrite = TRUE)

if (!is.null(alluvial_mask)) {
  writeRaster(alluvial_mask, file.path(out_dir, "alluvial_or_floodplain_support_mask.tif"), overwrite = TRUE)
}

if (split_upland) {
  writeRaster(zones_4, file.path(out_dir, "zones_4class_optionB_vectorRiver_valley_uplandSlope.tif"), overwrite = TRUE)
}

write.csv(zone_summary, file.path(out_dir, "zone_summary_optionB.csv"), row.names = FALSE)
write.csv(sensitivity_table, file.path(out_dir, "optionB_threshold_sensitivity_area_table.csv"), row.names = FALSE)
write.csv(metadata, file.path(out_dir, "run_metadata_optionB.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "R_sessionInfo_optionB.txt"))

# -----------------------------
# 11) QUICK PLOTS
# -----------------------------
png(file.path(out_dir, "quicklook_optionB_vector_river_valley_workflow.png"), width = 2100, height = 1400, res = 150)
par(mfrow = c(2, 3), mar = c(3, 3, 3, 5))
plot(dem, main = "Original DEM")
plot(dem_valley, main = "DEM used for valley predictors")
plot(slope_deg, main = "Slope for valley predictors (degrees)")
plot(tpi_large, main = "Large-window TPI")
plot(dist_to_river, main = "Distance to vector river (m)")
plot(zones_3, main = "Option B zones: 1 river, 2 valley, 3 upland")
dev.off()

png(file.path(out_dir, "quicklook_optionB_class2_only.png"), width = 1800, height = 1200, res = 150)
par(mfrow = c(1, 2), mar = c(3, 3, 3, 5))
plot(dist_to_river, main = "Distance to vector river (m)")
plot(valley, main = "Class 2: rule-based valley/floodplain proxy")
dev.off()

message("Done. Outputs written to: ", normalizePath(out_dir, winslash = "/", mustWork = FALSE))
message("Main output: ", file.path(out_dir, "zones_3class_optionB_vectorRiver_valley_upland.tif"))
message("Sensitivity table: ", file.path(out_dir, "optionB_threshold_sensitivity_area_table.csv"))
