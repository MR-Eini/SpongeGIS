# ============================================================
# Option A: Calibrated river valley / floodplain model - v7 four-class map
# ============================================================
# Purpose
#   Replace the weak DEM-only class 2 = valley/floodplain rule
#   with a calibrated model based on manual polygons and vector rivers.
#
# Main idea
#   1) Build terrain predictors from a DEM/rDTM.
#   2) Anchor the valley/floodplain search around vector rivers.
#   3) Calibrate candidate threshold combinations against manually
#      delineated polygons.
#   4) Select the best candidate using F1 / IoU / area bias.
#   5) Produce a final 4-class map:
#        1 = vector-river corridor / channel proxy
#        2 = calibrated river valley / floodplain
#        3 = upland / low-to-moderate-slope remaining terrain
#        4 = steep slope terrain (> threshold)
#
# Required packages
#   terra
#
# Optional package
#   whitebox: used only if you want to try HAND-like
#             height-above-stream calculation.
#
# Notes
#   - The method assumes projected coordinates in meters.
#   - The manual polygon layer should represent the target class 2.
#   - The alluvial soil layer is optional.
#   - Threshold calibration is first performed on a sample, then
#     exact raster metrics are computed for the top candidates.
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
# Adjust these paths before running.
dem_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/DEM50.tif"
river_vector_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/River/River.shp"
manual_valley_polygon_file <- "X:/download/Mamad/!SpongeGIStest/Kamienna/input/floodPlain_podtop500k/floodPlain_podtop500_k.shp"

# Optional supporting layer.
# Can be a raster or vector layer. Use NA_character_ if unavailable.
alluvial_soil_file <- NA_character_

# Output folder
out_dir <- "X:/download/Mamad/!SpongeGIStest/Kamienna/paper_zone/OptionA_calibrated_valley_floodplain_4zones"

# DEM smoothing for valley/floodplain terrain predictors.
# This helps reduce the influence of small artificial features such as embankments.
smooth_dem_for_valley <- TRUE
smooth_window_cells <- 5  # must be odd; 3 or 5 is usually enough

# TPI neighborhood sizes in meters.
# tpi_large is the main one used in calibration.
tpi_small_window_m <- 250
tpi_large_window_m <- 1000

# Vector-river corridor for class 1.
# This is not the floodplain. It is only a narrow channel/corridor proxy.
channel_buffer_m <- 50

# Final class 4 threshold.
# Class 4 is assigned after class 1 and class 2, so steep cells inside the
# vector-river corridor remain class 1, and calibrated floodplain/valley-floor
# cells remain class 2.
steep_slope_threshold_deg <- 5

# Candidate threshold values for calibration.
# Start broad, then narrow these ranges after inspecting the first results.
dist_thresholds_m <- c(50, 100, 150, 200, 300, 500, 750, 1000)
slope_thresholds_deg <- c(1, 2, 3, 5, 8, 12)
tpi_thresholds_m <- c(-3, -2, -1, 0, 1, 2)

# Optional HAND-like height-above-stream threshold.
# If Whitebox HAND calculation fails or is disabled, this is ignored.
use_whitebox_hand_if_available <- FALSE
hand_thresholds_m_if_available <- c(NA, 1, 2, 3, 5, 10)

# Optional alluvial-soil modes:
#   ignore  = do not use alluvial soils
#   require = candidate cells must also be alluvial
#   expand  = use alluvial soils to expand the candidate valley zone near rivers
# If alluvial_soil_file is NA, only "ignore" is used.
alluvial_modes_if_available <- c("ignore", "require", "expand")
alluvial_expand_slope_deg <- 5

# Calibration strategy.
# sample_n is used to screen many candidates quickly.
# Exact raster metrics are then computed for the best candidates.
calibration_sample_n <- 500000
# Stratified sampling avoids missing small manual floodplain polygons in a random sample.
# 0.5 means approximately half positive/reference cells and half negative/background cells.
calibration_positive_fraction <- 0.50
exact_top_n <- 30
save_top_candidate_rasters <- TRUE
save_top_n_rasters <- 5

# Majority filter for final class-2 map.
# Usually keep FALSE until the calibrated map has been inspected.
apply_majority_filter_to_valley <- FALSE
majority_window_size <- 3

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


inspect_vector_file <- function(file, layer_name, out_dir) {
  file_dir <- dirname(file)
  file_base <- tools::file_path_sans_ext(basename(file))
  sidecars <- list.files(
    file_dir,
    pattern = paste0("^", file_base, "\\."),
    full.names = TRUE
  )
  
  if (length(sidecars) > 0) {
    fi <- file.info(sidecars)
    sidecar_info <- data.frame(
      layer_name = layer_name,
      file = normalizePath(sidecars, winslash = "/", mustWork = FALSE),
      size_bytes = fi$size,
      modified_time = as.character(fi$mtime),
      stringsAsFactors = FALSE
    )
  } else {
    sidecar_info <- data.frame(
      layer_name = layer_name,
      file = normalizePath(file, winslash = "/", mustWork = FALSE),
      size_bytes = NA_real_,
      modified_time = NA_character_,
      stringsAsFactors = FALSE
    )
  }
  
  sidecar_diag_file <- file.path(out_dir, paste0(layer_name, "_vector_file_sidecars.csv"))
  write.csv(sidecar_info, sidecar_diag_file, row.names = FALSE)
  
  vect_diag <- data.frame(
    layer_name = layer_name,
    file = normalizePath(file, winslash = "/", mustWork = FALSE),
    read_success = FALSE,
    n_features = NA_integer_,
    geomtype = NA_character_,
    crs = NA_character_,
    xmin = NA_real_, xmax = NA_real_, ymin = NA_real_, ymax = NA_real_,
    message = NA_character_,
    stringsAsFactors = FALSE
  )
  
  v_try <- try(terra::vect(file), silent = TRUE)
  if (inherits(v_try, "try-error")) {
    vect_diag$message <- paste(as.character(v_try), collapse = " ")
  } else {
    vect_diag$read_success <- TRUE
    vect_diag$n_features <- nrow(v_try)
    vect_diag$geomtype <- paste(unique(as.character(terra::geomtype(v_try))), collapse = ";")
    vect_diag$crs <- terra::crs(v_try)
    e <- try(terra::ext(v_try), silent = TRUE)
    if (!inherits(e, "try-error")) {
      vect_diag$xmin <- e$xmin; vect_diag$xmax <- e$xmax
      vect_diag$ymin <- e$ymin; vect_diag$ymax <- e$ymax
    }
    vect_diag$message <- ifelse(nrow(v_try) == 0, "Layer reads successfully but contains zero features.", "Layer contains features.")
  }
  
  vector_diag_file <- file.path(out_dir, paste0(layer_name, "_vector_read_diagnostics.csv"))
  write.csv(vect_diag, vector_diag_file, row.names = FALSE)
  
  list(sidecar_diag_file = sidecar_diag_file, vector_diag_file = vector_diag_file)
}

make_circular_window <- function(radius_m, res_m) {
  radius_cells <- ceiling(radius_m / res_m)
  radius_cells <- max(radius_cells, 1)
  idx <- -radius_cells:radius_cells
  w <- outer(idx, idx, function(i, j) sqrt(i^2 + j^2) <= radius_cells)
  w <- matrix(as.numeric(w), nrow = length(idx), ncol = length(idx))
  w[w == 0] <- NA_real_
  w
}

safe_focal_mean <- function(x, window_m, res_m) {
  w <- make_circular_window(window_m, res_m)
  focal(x, w = w, fun = mean, na.policy = "omit", fillvalue = NA)
}

raster_sum <- function(x) {
  val <- terra::global(x, "sum", na.rm = TRUE)[1, 1]
  as.numeric(val)
}

metrics_from_counts <- function(tp, fp, fn, tn) {
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), NA_real_)
  recall    <- ifelse((tp + fn) > 0, tp / (tp + fn), NA_real_)
  f1        <- ifelse(!is.na(precision + recall) && (precision + recall) > 0,
                      2 * precision * recall / (precision + recall), NA_real_)
  iou       <- ifelse((tp + fp + fn) > 0, tp / (tp + fp + fn), NA_real_)
  accuracy  <- ifelse((tp + fp + fn + tn) > 0, (tp + tn) / (tp + fp + fn + tn), NA_real_)
  pred_pos  <- tp + fp
  ref_pos   <- tp + fn
  area_bias_ratio <- ifelse(ref_pos > 0, pred_pos / ref_pos, NA_real_)
  area_bias_pct   <- ifelse(!is.na(area_bias_ratio), 100 * (area_bias_ratio - 1), NA_real_)
  
  data.frame(
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn,
    precision = precision,
    recall = recall,
    f1 = f1,
    iou = iou,
    accuracy = accuracy,
    pred_positive_cells = pred_pos,
    reference_positive_cells = ref_pos,
    area_bias_ratio = area_bias_ratio,
    area_bias_pct = area_bias_pct
  )
}

metrics_from_vectors <- function(pred, ref) {
  pred <- as.logical(pred)
  ref  <- as.logical(ref)
  keep <- !is.na(pred) & !is.na(ref)
  pred <- pred[keep]
  ref  <- ref[keep]
  
  tp <- sum(pred & ref)
  fp <- sum(pred & !ref)
  fn <- sum(!pred & ref)
  tn <- sum(!pred & !ref)
  metrics_from_counts(tp, fp, fn, tn)
}

metrics_from_rasters <- function(pred01, ref01, valid) {
  pred01 <- ifel(is.na(pred01), 0, pred01)
  ref01  <- ifel(is.na(ref01),  0, ref01)
  
  tp <- raster_sum(ifel(valid & (pred01 == 1) & (ref01 == 1), 1, 0))
  fp <- raster_sum(ifel(valid & (pred01 == 1) & (ref01 == 0), 1, 0))
  fn <- raster_sum(ifel(valid & (pred01 == 0) & (ref01 == 1), 1, 0))
  tn <- raster_sum(ifel(valid & (pred01 == 0) & (ref01 == 0), 1, 0))
  
  metrics_from_counts(tp, fp, fn, tn)
}

extent_overlap <- function(a, b) {
  ea <- terra::ext(a)
  eb <- terra::ext(b)
  !(terra::xmax(ea) < terra::xmin(eb) ||
      terra::xmin(ea) > terra::xmax(eb) ||
      terra::ymax(ea) < terra::ymin(eb) ||
      terra::ymin(ea) > terra::ymax(eb))
}

extent_to_df <- function(x, label) {
  e <- terra::ext(x)
  data.frame(
    layer = label,
    xmin = terra::xmin(e),
    xmax = terra::xmax(e),
    ymin = terra::ymin(e),
    ymax = terra::ymax(e),
    crs = terra::crs(x),
    stringsAsFactors = FALSE
  )
}

sample_predictors_from_mask <- function(predictor_layers, mask_logical, size, label) {
  size <- as.integer(size)
  if (is.na(size) || size <= 0) return(data.frame())
  
  # Sample coordinates from the mask, not from the predictor stack.
  # This keeps positive reference cells even if some predictors are NA.
  mask_r <- ifel(mask_logical, 1, NA)
  
  pts <- terra::spatSample(
    mask_r,
    size = size,
    method = "random",
    as.df = TRUE,
    xy = TRUE,
    values = FALSE,
    na.rm = TRUE
  )
  
  if (is.null(pts) || nrow(pts) == 0) {
    warning("No sampled points were returned for ", label, ".")
    return(data.frame())
  }
  
  vals <- terra::extract(predictor_layers, as.matrix(pts[, c("x", "y")]))
  vals <- vals[, setdiff(names(vals), "ID"), drop = FALSE]
  
  out <- cbind(sample_group = label, pts[, c("x", "y"), drop = FALSE], vals)
  out
}

write_spatial_diagnostics <- function(template, original_layer, projected_layer, layer_name, out_dir) {
  diag <- rbind(
    extent_to_df(template, "DEM_template"),
    extent_to_df(original_layer, paste0(layer_name, "_original")),
    extent_to_df(projected_layer, paste0(layer_name, "_projected_to_DEM"))
  )
  diag$overlaps_DEM_extent <- c(
    TRUE,
    extent_overlap(original_layer, template),
    extent_overlap(projected_layer, template)
  )
  diag_file <- file.path(out_dir, paste0(layer_name, "_spatial_diagnostics.csv"))
  write.csv(diag, diag_file, row.names = FALSE)
  diag_file
}

load_binary_layer <- function(file, template, layer_name, required_overlap = TRUE) {
  if (is.na(file) || !nzchar(file)) return(NULL)
  stop_if_missing(file, layer_name)
  
  ext <- tolower(tools::file_ext(file))
  raster_exts <- c("tif", "tiff", "img", "grd", "asc")
  
  if (ext %in% raster_exts) {
    r0 <- rast(file)
    if (terra::crs(r0) == "") {
      stop(layer_name, " has no CRS: ", file)
    }
    
    r <- r0
    if (terra::crs(r) != terra::crs(template)) {
      r <- terra::project(r, template, method = "near")
    }
    
    diag_file <- write_spatial_diagnostics(template, r0, r, layer_name, out_dir)
    
    if (!extent_overlap(r, template)) {
      msg <- paste0(
        layer_name, " raster extent does not overlap the DEM extent after CRS handling. ",
        "Diagnostic file written to: ", diag_file, ". ",
        "This usually means wrong CRS, wrong input layer, or a layer outside the DEM extent."
      )
      if (required_overlap) stop(msg) else warning(msg)
    }
    
    if (!terra::compareGeom(r, template, stopOnError = FALSE)) {
      r <- terra::resample(r, template, method = "near")
    }
    r <- ifel(is.na(r), 0, ifel(r > 0, 1, 0))
  } else {
    v0 <- vect(file)
    
    if (nrow(v0) == 0) {
      diag_files <- inspect_vector_file(file, layer_name, out_dir)
      stop(
        layer_name, " vector has zero features: ", file, "\n",
        "This means the file exists and can be opened, but it contains no records. ",
        "Likely causes: wrong shapefile, empty export, no features selected during export, ",
        "or missing/corrupted shapefile sidecar files.\n",
        "Diagnostics written to:\n  - ", diag_files$vector_diag_file, "\n  - ", diag_files$sidecar_diag_file, "\n",
        "Open the layer in QGIS/ArcGIS and check the attribute table feature count."
      )
    }
    if (terra::crs(v0) == "") {
      stop(layer_name, " has no CRS. Define the correct CRS before running: ", file)
    }
    
    v <- v0
    if (terra::crs(v) != terra::crs(template)) {
      v <- terra::project(v, terra::crs(template))
    }
    
    # Try to repair invalid polygon geometries if terra supports it.
    if ("makeValid" %in% getNamespaceExports("terra")) {
      v <- tryCatch(terra::makeValid(v), error = function(e) v)
    }
    
    diag_file <- write_spatial_diagnostics(template, v0, v, layer_name, out_dir)
    
    if (nrow(v) == 0) {
      stop(
        layer_name, " vector has zero features after projection/geometry handling. ",
        "Diagnostic file written to: ", diag_file
      )
    }
    
    if (!extent_overlap(v, template)) {
      msg <- paste0(
        layer_name, " vector extent does not overlap the DEM extent after CRS handling. ",
        "Diagnostic file written to: ", diag_file, ". ",
        "Do not continue calibration until this is fixed. Likely causes: ",
        "(1) wrong CRS assigned to the shapefile, (2) wrong shapefile, ",
        "(3) DEM and vector are from different study areas, or ",
        "(4) coordinates are stored in one CRS but labelled as another."
      )
      if (required_overlap) stop(msg) else warning(msg)
    }
    
    # Crop only after confirming extent overlap. This improves rasterization speed.
    v_crop <- tryCatch(terra::crop(v, terra::ext(template)), error = function(e) v)
    if (nrow(v_crop) == 0) {
      msg <- paste0(
        layer_name, " has extent overlap but no features remain after cropping to DEM extent. ",
        "This can happen when only bounding boxes touch, or when geometries are invalid. ",
        "Diagnostic file written to: ", diag_file
      )
      if (required_overlap) stop(msg) else warning(msg)
      v_crop <- v
    }
    
    # Robust binary rasterization. Burn a constant value of 1 into the DEM grid.
    r <- terra::rasterize(v_crop, template, field = 1, background = 0, touches = TRUE)
    r <- ifel(is.na(r), 0, ifel(r > 0, 1, 0))
  }
  
  names(r) <- layer_name
  r
}

candidate_predict_df <- function(df,
                                 dist_threshold_m,
                                 slope_threshold_deg,
                                 tpi_threshold_m,
                                 hand_threshold_m,
                                 alluvial_mode,
                                 alluvial_expand_slope_deg = 5) {
  pred <- !is.na(df$dist_to_river_m) &
    !is.na(df$slope_deg) &
    !is.na(df$tpi_large) &
    df$dist_to_river_m <= dist_threshold_m &
    df$slope_deg <= slope_threshold_deg &
    df$tpi_large <= tpi_threshold_m
  
  if (!is.na(hand_threshold_m) && "hand_m" %in% names(df)) {
    pred <- pred & !is.na(df$hand_m) & df$hand_m <= hand_threshold_m
  }
  
  if (alluvial_mode == "require" && "alluvial" %in% names(df)) {
    pred <- pred & !is.na(df$alluvial) & df$alluvial == 1
  }
  
  if (alluvial_mode == "expand" && "alluvial" %in% names(df)) {
    expanded <- !is.na(df$dist_to_river_m) &
      !is.na(df$slope_deg) &
      !is.na(df$alluvial) &
      df$dist_to_river_m <= dist_threshold_m &
      df$slope_deg <= alluvial_expand_slope_deg &
      df$alluvial == 1
    pred <- pred | expanded
  }
  
  pred[is.na(pred)] <- FALSE
  pred
}

candidate_predict_raster <- function(dist_to_river_m,
                                     slope_deg,
                                     tpi_large,
                                     valid,
                                     dist_threshold_m,
                                     slope_threshold_deg,
                                     tpi_threshold_m,
                                     hand_threshold_m = NA_real_,
                                     hand_m = NULL,
                                     alluvial_mode = "ignore",
                                     alluvial = NULL,
                                     alluvial_expand_slope_deg = 5) {
  pred <- valid &
    (dist_to_river_m <= dist_threshold_m) &
    (slope_deg <= slope_threshold_deg) &
    (tpi_large <= tpi_threshold_m)
  
  if (!is.na(hand_threshold_m) && !is.null(hand_m)) {
    pred <- pred & (hand_m <= hand_threshold_m)
  }
  
  if (alluvial_mode == "require" && !is.null(alluvial)) {
    pred <- pred & (alluvial == 1)
  }
  
  if (alluvial_mode == "expand" && !is.null(alluvial)) {
    expanded <- valid &
      (dist_to_river_m <= dist_threshold_m) &
      (slope_deg <= alluvial_expand_slope_deg) &
      (alluvial == 1)
    pred <- pred | expanded
  }
  
  ifel(pred, 1, 0)
}

rank_metrics <- function(x) {
  # Higher F1 and IoU are better; lower absolute area bias is better.
  x$abs_area_bias_pct <- abs(x$area_bias_pct)
  x[order(-x$f1, -x$iou, x$abs_area_bias_pct, -x$recall), ]
}

zonal_mean_table <- function(r, zones, var_name) {
  z <- zonal(r, zones, fun = "mean", na.rm = TRUE)
  colnames(z) <- c("zone", paste0(var_name, "_mean"))
  z
}

# Robust summary helpers.
# These avoid terra::freq() column-name differences between terra versions.
# They also keep all classes 1, 2, 3, and 4 even if one class is absent.
zone_count_table <- function(zones, valid_zone_values = c(1L, 2L, 3L, 4L)) {
  zdf <- terra::as.data.frame(zones, na.rm = TRUE)
  if (is.null(zdf) || nrow(zdf) == 0) {
    stop("No non-NA cells found in the zones raster.")
  }
  zone_vals <- zdf[[1]]
  zone_vals <- zone_vals[!is.na(zone_vals)]
  zone_vals <- as.integer(round(zone_vals))
  zone_vals <- zone_vals[zone_vals %in% valid_zone_values]
  if (length(zone_vals) == 0) {
    stop("No valid class values 1, 2, 3, or 4 found in the zones raster.")
  }
  tab <- table(factor(zone_vals, levels = valid_zone_values))
  data.frame(zone = as.integer(names(tab)), n_cells = as.numeric(tab), stringsAsFactors = FALSE)
}

robust_zonal_stat <- function(r, zones, var_name, stat_name) {
  out_col <- paste0(var_name, "_", stat_name)
  z <- try(as.data.frame(terra::zonal(r, zones, fun = stat_name, na.rm = TRUE)), silent = TRUE)
  if (inherits(z, "try-error") || is.null(z) || nrow(z) == 0 || ncol(z) < 2) {
    warning("Zonal statistic failed or returned empty output for ", out_col, ". Filling with NA.")
    out <- data.frame(zone = c(1L, 2L, 3L, 4L), tmp = NA_real_, stringsAsFactors = FALSE)
    names(out) <- c("zone", out_col)
    return(out)
  }
  nms <- names(z)
  possible_zone_cols <- c("zone", names(zones)[1], "value")
  zone_col <- possible_zone_cols[possible_zone_cols %in% nms][1]
  if (is.na(zone_col) || !nzchar(zone_col)) zone_col <- nms[1]
  value_cols <- setdiff(nms, zone_col)
  value_col <- value_cols[length(value_cols)]
  out <- data.frame(zone = as.integer(round(z[[zone_col]])), tmp = as.numeric(z[[value_col]]), stringsAsFactors = FALSE)
  names(out)[2] <- out_col
  out[out$zone %in% c(1L, 2L, 3L, 4L), , drop = FALSE]
}

make_zone_summary_robust <- function(zones, elevation_r, slope_deg_r, slope_pct_r, tpi_large_r, dist_to_river_r) {
  message("Building robust final zone summary...")
  valid_zone_values <- c(1L, 2L, 3L, 4L)
  cell_area_m2_local <- prod(terra::res(zones))
  zone_names <- data.frame(
    zone = valid_zone_values,
    zone_name = c(
      "channel_vector_corridor",
      "calibrated_valley_floodplain_proxy",
      "upland_low_to_moderate_slope",
      "steep_slope_gt_threshold"
    ),
    stringsAsFactors = FALSE
  )
  counts <- zone_count_table(zones, valid_zone_values)
  counts$area_m2 <- counts$n_cells * cell_area_m2_local
  counts$area_km2 <- counts$area_m2 / 1e6
  total_area_m2 <- sum(counts$area_m2, na.rm = TRUE)
  if (is.finite(total_area_m2) && total_area_m2 > 0) {
    counts$area_percent <- 100 * counts$area_m2 / total_area_m2
  } else {
    counts$area_percent <- NA_real_
  }
  stat_tables <- list(
    robust_zonal_stat(elevation_r,     zones, "elevation",           "mean"),
    robust_zonal_stat(slope_deg_r,     zones, "slope_deg",           "mean"),
    robust_zonal_stat(slope_deg_r,     zones, "slope_deg",           "min"),
    robust_zonal_stat(slope_deg_r,     zones, "slope_deg",           "max"),
    robust_zonal_stat(slope_pct_r,     zones, "slope_pct",           "mean"),
    robust_zonal_stat(tpi_large_r,     zones, "tpi_large",           "mean"),
    robust_zonal_stat(dist_to_river_r, zones, "distance_to_river_m", "mean")
  )
  out <- merge(zone_names, counts, by = "zone", all.x = TRUE)
  for (st in stat_tables) out <- merge(out, st, by = "zone", all.x = TRUE)
  desired_cols <- c("zone", "zone_name", "n_cells", "area_m2", "area_km2", "area_percent",
                    "elevation_mean", "slope_deg_mean", "slope_deg_min", "slope_deg_max",
                    "slope_pct_mean", "tpi_large_mean", "distance_to_river_m_mean")
  for (cc in desired_cols) if (!cc %in% names(out)) out[[cc]] <- NA
  out <- out[, desired_cols]
  names(out)[names(out) == "distance_to_river_m_mean"] <- "distance_to_river_mean_m"
  out[order(out$zone), ]
}

# -----------------------------
# 3) BASIC CHECKS AND OUTPUTS
# -----------------------------
stop_if_missing(dem_file, "DEM file")
stop_if_missing(river_vector_file, "River vector file")
stop_if_missing(manual_valley_polygon_file, "Manual valley/floodplain polygon file")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

message("Reading DEM...")
dem <- rast(dem_file)

if (terra::crs(dem) == "") {
  stop("The DEM has no CRS. A projected CRS in meters is required.")
}
if (terra::is.lonlat(dem)) {
  stop("The DEM is in longitude/latitude degrees. Reproject it to a projected CRS in meters before running this workflow.")
}

res_xy <- res(dem)
if (abs(res_xy[1] - res_xy[2]) > 1e-6) {
  warning("DEM cells are not square. The first resolution value will be used for window-size conversion.")
}
res_m <- res_xy[1]
cell_area_m2 <- prod(res_xy)
valid <- !is.na(dem)

# -----------------------------
# 4) DEM PREPARATION AND TERRAIN PREDICTORS
# -----------------------------
message("Preparing DEM-derived predictors...")

if (smooth_dem_for_valley) {
  smooth_window_cells <- make_odd(smooth_window_cells)
  w_smooth <- matrix(1, smooth_window_cells, smooth_window_cells)
  dem_geom <- focal(dem, w = w_smooth, fun = mean, na.policy = "omit", fillvalue = NA)
  names(dem_geom) <- "dem_smoothed"
} else {
  dem_geom <- dem
  names(dem_geom) <- "dem_for_geomorphometry"
}

slope_deg <- terrain(dem_geom, v = "slope", unit = "degrees")
names(slope_deg) <- "slope_deg"

slope_pct <- tan(slope_deg * pi / 180) * 100
names(slope_pct) <- "slope_pct"

tpi_small <- dem_geom - safe_focal_mean(dem_geom, tpi_small_window_m, res_m)
names(tpi_small) <- "tpi_small"

tpi_large <- dem_geom - safe_focal_mean(dem_geom, tpi_large_window_m, res_m)
names(tpi_large) <- "tpi_large"

writeRaster(dem_geom, file.path(out_dir, "dem_used_for_valley_predictors.tif"), overwrite = TRUE)
writeRaster(slope_deg, file.path(out_dir, "predictor_slope_deg.tif"), overwrite = TRUE)
writeRaster(slope_pct, file.path(out_dir, "predictor_slope_pct.tif"), overwrite = TRUE)
writeRaster(tpi_small, file.path(out_dir, "predictor_tpi_small.tif"), overwrite = TRUE)
writeRaster(tpi_large, file.path(out_dir, "predictor_tpi_large.tif"), overwrite = TRUE)

# -----------------------------
# 5) VECTOR RIVERS: RASTERIZE AND DISTANCE
# -----------------------------
message("Preparing vector-river distance layer...")
rivers_original <- vect(river_vector_file)
if (nrow(rivers_original) == 0) {
  stop("River vector has zero features: ", river_vector_file)
}
if (terra::crs(rivers_original) == "") {
  stop("River vector has no CRS. Define the correct CRS before running: ", river_vector_file)
}

rivers <- rivers_original
if (terra::crs(rivers) != terra::crs(dem)) {
  rivers <- terra::project(rivers, terra::crs(dem))
}

river_diag_file <- write_spatial_diagnostics(dem, rivers_original, rivers, "river_vector", out_dir)
if (!extent_overlap(rivers, dem)) {
  stop(
    "River vector extent does not overlap the DEM after CRS handling. ",
    "Diagnostic file written to: ", river_diag_file, ". ",
    "Check whether the river layer has the correct CRS and study area."
  )
}

rivers_crop <- tryCatch(terra::crop(rivers, terra::ext(dem)), error = function(e) rivers)
if (nrow(rivers_crop) == 0) {
  stop(
    "River vector overlaps the DEM extent, but no river features remain after cropping. ",
    "Diagnostic file written to: ", river_diag_file
  )
}

river_raster <- terra::rasterize(rivers_crop, dem, field = 1, background = NA, touches = TRUE)
names(river_raster) <- "river_cells"

river_cell_count <- raster_sum(ifel(valid & !is.na(river_raster), 1, 0))
if (river_cell_count == 0) {
  stop(
    "River vector rasterization produced zero river cells. ",
    "Check river_vector_spatial_diagnostics.csv and the river layer geometry."
  )
}
message("Rasterized river cells: ", river_cell_count)

dist_to_river_m <- terra::distance(river_raster)
names(dist_to_river_m) <- "dist_to_river_m"

writeRaster(river_raster, file.path(out_dir, "river_rasterized.tif"), overwrite = TRUE)
writeRaster(dist_to_river_m, file.path(out_dir, "predictor_distance_to_vector_river_m.tif"), overwrite = TRUE)

# -----------------------------
# 6) MANUAL POLYGONS AS REFERENCE CLASS 2
# -----------------------------
message("Preparing manual valley/floodplain reference raster...")
manual_ref <- load_binary_layer(manual_valley_polygon_file, dem, "manual_ref", required_overlap = TRUE)
writeRaster(manual_ref, file.path(out_dir, "reference_manual_valley_floodplain.tif"), overwrite = TRUE, datatype = "INT1U")

manual_positive_full <- raster_sum(ifel(valid & (manual_ref == 1), 1, 0))
manual_negative_full <- raster_sum(ifel(valid & (manual_ref == 0), 1, 0))
message("Manual reference positive cells in full DEM grid: ", manual_positive_full)
message("Manual reference negative cells in full DEM grid: ", manual_negative_full)

# CRS/extent diagnostics for troubleshooting manual-reference alignment.
manual_ref_vect_debug <- vect(manual_valley_polygon_file)
if (terra::crs(manual_ref_vect_debug) != "" && terra::crs(manual_ref_vect_debug) != terra::crs(dem)) {
  manual_ref_vect_debug <- terra::project(manual_ref_vect_debug, terra::crs(dem))
}
ref_diag <- rbind(
  extent_to_df(dem, "DEM"),
  extent_to_df(rivers, "river_vector_projected"),
  extent_to_df(manual_ref_vect_debug, "manual_polygon_projected_or_original")
)
write.csv(ref_diag, file.path(out_dir, "input_crs_extent_diagnostics.csv"), row.names = FALSE)

if (manual_positive_full == 0) {
  stop(
    "The manual polygon layer produced zero positive cells after rasterization. ",
    "Open input_crs_extent_diagnostics.csv and reference_manual_valley_floodplain.tif. ",
    "Most likely causes: wrong CRS assigned to the polygon layer, no spatial overlap with DEM, ",
    "or polygons are outside the DEM extent."
  )
}

# -----------------------------
# 7) OPTIONAL ALLUVIAL SOIL LAYER
# -----------------------------
message("Preparing optional alluvial-soil layer...")
alluvial <- load_binary_layer(alluvial_soil_file, dem, "alluvial", required_overlap = FALSE)
has_alluvial <- !is.null(alluvial)
if (has_alluvial) {
  writeRaster(alluvial, file.path(out_dir, "predictor_alluvial_soils_binary.tif"), overwrite = TRUE)
}

# -----------------------------
# 8) OPTIONAL HAND-LIKE LAYER WITH WHITEBOX
# -----------------------------
hand_m <- NULL
has_hand <- FALSE

if (isTRUE(use_whitebox_hand_if_available)) {
  message("Trying optional Whitebox HAND-like elevation-above-stream calculation...")
  if (requireNamespace("whitebox", quietly = TRUE) &&
      "wbt_elevation_above_stream" %in% getNamespaceExports("whitebox")) {
    dem_wbt_file <- file.path(out_dir, "dem_for_whitebox_hand.tif")
    stream_wbt_file <- file.path(out_dir, "river_stream_for_whitebox_hand.tif")
    hand_wbt_file <- file.path(out_dir, "predictor_hand_whitebox_m.tif")
    
    stream01 <- ifel(is.na(river_raster), 0, 1)
    writeRaster(dem, dem_wbt_file, overwrite = TRUE)
    writeRaster(stream01, stream_wbt_file, overwrite = TRUE, datatype = "INT1U")
    
    hand_try <- try(
      whitebox::wbt_elevation_above_stream(
        dem = dem_wbt_file,
        streams = stream_wbt_file,
        output = hand_wbt_file
      ),
      silent = TRUE
    )
    
    if (!inherits(hand_try, "try-error") && file.exists(hand_wbt_file)) {
      hand_m <- rast(hand_wbt_file)
      if (!terra::compareGeom(hand_m, dem, stopOnError = FALSE)) {
        hand_m <- terra::resample(hand_m, dem, method = "bilinear")
      }
      names(hand_m) <- "hand_m"
      has_hand <- TRUE
    } else {
      message("Whitebox HAND calculation failed; continuing without HAND.")
    }
  } else {
    message("Whitebox or wbt_elevation_above_stream() is unavailable; continuing without HAND.")
  }
}

# -----------------------------
# 9) BUILD CALIBRATION SAMPLE
# -----------------------------
message("Building stratified calibration sample...")

predictor_layers <- c(dist_to_river_m, slope_deg, tpi_small, tpi_large, manual_ref)
if (has_hand) predictor_layers <- c(predictor_layers, hand_m)
if (has_alluvial) predictor_layers <- c(predictor_layers, alluvial)

# Ensure stable names.
names(predictor_layers) <- make.unique(names(predictor_layers))

# Do not use a purely random sample here. If the manual floodplain polygons are
# small relative to the DEM, a random sample may contain zero positive cells.
# Instead, sample positives and negatives separately.
set.seed(123)

n_pos_target <- min(
  manual_positive_full,
  max(1, floor(calibration_sample_n * calibration_positive_fraction))
)
n_neg_target <- min(
  manual_negative_full,
  max(1, calibration_sample_n - n_pos_target)
)

message("Target positive sample cells: ", n_pos_target)
message("Target negative sample cells: ", n_neg_target)

pos_mask <- valid & (manual_ref == 1)
neg_mask <- valid & (manual_ref == 0)

pos_df <- sample_predictors_from_mask(
  predictor_layers = predictor_layers,
  mask_logical = pos_mask,
  size = n_pos_target,
  label = "manual_positive"
)

neg_df <- sample_predictors_from_mask(
  predictor_layers = predictor_layers,
  mask_logical = neg_mask,
  size = n_neg_target,
  label = "manual_negative"
)

calib_df <- rbind(pos_df, neg_df)

if (nrow(calib_df) == 0) {
  stop("No calibration rows were sampled. Check DEM validity and reference rasterization.")
}

# Keep only rows with a valid manual reference.
calib_df <- calib_df[!is.na(calib_df$manual_ref), ]
calib_df$manual_ref <- ifelse(calib_df$manual_ref > 0, 1, 0)

message("Calibration sample rows: ", nrow(calib_df))
message("Manual reference positive cells in sample: ", sum(calib_df$manual_ref == 1))
message("Manual reference negative cells in sample: ", sum(calib_df$manual_ref == 0))

write.csv(
  data.frame(
    item = c("full_positive_cells", "full_negative_cells", "sample_positive_cells", "sample_negative_cells"),
    value = c(
      manual_positive_full,
      manual_negative_full,
      sum(calib_df$manual_ref == 1),
      sum(calib_df$manual_ref == 0)
    )
  ),
  file.path(out_dir, "manual_reference_sampling_diagnostics.csv"),
  row.names = FALSE
)

if (sum(calib_df$manual_ref == 1) == 0) {
  stop(
    "No positive manual valley/floodplain cells were found in the stratified calibration sample, ",
    "although full-raster positives were counted. This usually means predictor extraction failed for positive cells. ",
    "Check manual_reference_sampling_diagnostics.csv and input_crs_extent_diagnostics.csv."
  )
}

# -----------------------------
# 10) CANDIDATE CALIBRATION GRID
# -----------------------------
alluvial_modes <- "ignore"
if (has_alluvial) alluvial_modes <- alluvial_modes_if_available

hand_thresholds_m <- NA_real_
if (has_hand) hand_thresholds_m <- hand_thresholds_m_if_available

candidate_grid <- expand.grid(
  dist_threshold_m = dist_thresholds_m,
  slope_threshold_deg = slope_thresholds_deg,
  tpi_threshold_m = tpi_thresholds_m,
  hand_threshold_m = hand_thresholds_m,
  alluvial_mode = alluvial_modes,
  stringsAsFactors = FALSE
)

candidate_grid$candidate_id <- seq_len(nrow(candidate_grid))
candidate_grid <- candidate_grid[, c("candidate_id", "dist_threshold_m", "slope_threshold_deg", "tpi_threshold_m", "hand_threshold_m", "alluvial_mode")]

message("Number of candidate threshold combinations: ", nrow(candidate_grid))

# -----------------------------
# 11) SAMPLE-BASED CALIBRATION
# -----------------------------
message("Scoring candidates on calibration sample...")

sample_metrics_list <- vector("list", nrow(candidate_grid))
ref_vec <- calib_df$manual_ref == 1

for (i in seq_len(nrow(candidate_grid))) {
  cg <- candidate_grid[i, ]
  pred_vec <- candidate_predict_df(
    calib_df,
    dist_threshold_m = cg$dist_threshold_m,
    slope_threshold_deg = cg$slope_threshold_deg,
    tpi_threshold_m = cg$tpi_threshold_m,
    hand_threshold_m = cg$hand_threshold_m,
    alluvial_mode = cg$alluvial_mode,
    alluvial_expand_slope_deg = alluvial_expand_slope_deg
  )
  
  m <- metrics_from_vectors(pred_vec, ref_vec)
  sample_metrics_list[[i]] <- cbind(cg, m)
  
  if (i %% 100 == 0) message("  scored ", i, " / ", nrow(candidate_grid), " candidates")
}

sample_metrics <- do.call(rbind, sample_metrics_list)
sample_metrics_ranked <- rank_metrics(sample_metrics)
write.csv(sample_metrics_ranked, file.path(out_dir, "candidate_metrics_sample_ranked.csv"), row.names = FALSE)

# -----------------------------
# 12) EXACT RASTER METRICS FOR TOP CANDIDATES
# -----------------------------
message("Computing exact raster metrics for top candidates...")

top_candidates <- head(sample_metrics_ranked$candidate_id, exact_top_n)
exact_metrics_list <- vector("list", length(top_candidates))

for (j in seq_along(top_candidates)) {
  id <- top_candidates[j]
  cg <- candidate_grid[candidate_grid$candidate_id == id, ]
  
  pred_r <- candidate_predict_raster(
    dist_to_river_m = dist_to_river_m,
    slope_deg = slope_deg,
    tpi_large = tpi_large,
    valid = valid,
    dist_threshold_m = cg$dist_threshold_m,
    slope_threshold_deg = cg$slope_threshold_deg,
    tpi_threshold_m = cg$tpi_threshold_m,
    hand_threshold_m = cg$hand_threshold_m,
    hand_m = hand_m,
    alluvial_mode = cg$alluvial_mode,
    alluvial = alluvial,
    alluvial_expand_slope_deg = alluvial_expand_slope_deg
  )
  
  m <- metrics_from_rasters(pred_r, manual_ref, valid)
  exact_metrics_list[[j]] <- cbind(cg, m)
  
  if (save_top_candidate_rasters && j <= save_top_n_rasters) {
    out_name <- sprintf("top_%02d_candidate_%04d_valley_floodplain.tif", j, id)
    writeRaster(pred_r, file.path(out_dir, out_name), overwrite = TRUE, datatype = "INT1U")
  }
}

exact_metrics <- do.call(rbind, exact_metrics_list)
exact_metrics_ranked <- rank_metrics(exact_metrics)
write.csv(exact_metrics_ranked, file.path(out_dir, "candidate_metrics_exact_top_ranked.csv"), row.names = FALSE)

best <- exact_metrics_ranked[1, ]
message("Best candidate ID: ", best$candidate_id)
message("Best exact F1: ", round(best$f1, 4), "; IoU: ", round(best$iou, 4), "; area bias (%): ", round(best$area_bias_pct, 2))

# -----------------------------
# 13) FINAL BEST VALLEY/FLOODPLAIN MAP
# -----------------------------
message("Writing final calibrated valley/floodplain map...")

best_valley_raw <- candidate_predict_raster(
  dist_to_river_m = dist_to_river_m,
  slope_deg = slope_deg,
  tpi_large = tpi_large,
  valid = valid,
  dist_threshold_m = best$dist_threshold_m,
  slope_threshold_deg = best$slope_threshold_deg,
  tpi_threshold_m = best$tpi_threshold_m,
  hand_threshold_m = best$hand_threshold_m,
  hand_m = hand_m,
  alluvial_mode = best$alluvial_mode,
  alluvial = alluvial,
  alluvial_expand_slope_deg = alluvial_expand_slope_deg
)

names(best_valley_raw) <- "calibrated_valley_floodplain_raw"
writeRaster(best_valley_raw, file.path(out_dir, "calibrated_valley_floodplain_raw.tif"), overwrite = TRUE, datatype = "INT1U")

best_valley <- best_valley_raw

if (apply_majority_filter_to_valley) {
  mw <- matrix(1, make_odd(majority_window_size), make_odd(majority_window_size))
  best_valley <- focal(best_valley, w = mw, fun = modal, na.policy = "omit", fillvalue = NA)
  best_valley <- ifel(is.na(best_valley), 0, ifel(best_valley > 0, 1, 0))
  names(best_valley) <- "calibrated_valley_floodplain_filtered"
  writeRaster(best_valley, file.path(out_dir, "calibrated_valley_floodplain_filtered.tif"), overwrite = TRUE, datatype = "INT1U")
}

# -----------------------------
# 14) FINAL 4-CLASS MAP
# -----------------------------
message("Building final 4-class terrain-zone map...")

channel <- valid & (dist_to_river_m <= channel_buffer_m)
channel01 <- ifel(channel, 1, 0)

# Exclude the narrow river corridor from class 2.
valley_no_channel <- valid & (best_valley == 1) & (channel01 == 0)
valley01 <- ifel(valley_no_channel, 1, 0)

# Class 4: steep slopes above the user-defined threshold.
# Priority order is 1 -> 2 -> 4 -> 3:
#   - steep cells inside the vector-river corridor remain class 1;
#   - calibrated floodplain/valley-floor cells remain class 2;
#   - only the remaining terrain with slope > threshold becomes class 4.
steep_slope <- valid &
  (channel01 == 0) &
  (valley01 == 0) &
  (slope_deg > steep_slope_threshold_deg)
steep_slope01 <- ifel(steep_slope, 1, 0)

upland_low_to_moderate_slope <- valid &
  (channel01 == 0) &
  (valley01 == 0) &
  (steep_slope01 == 0)
upland_low_to_moderate_slope01 <- ifel(upland_low_to_moderate_slope, 1, 0)

zones_4 <- ifel(channel01 == 1, 1,
                ifel(valley01 == 1, 2,
                     ifel(steep_slope01 == 1, 4,
                          ifel(valid, 3, NA))))
names(zones_4) <- "zone"

writeRaster(channel01, file.path(out_dir, "class1_vector_river_corridor.tif"), overwrite = TRUE, datatype = "INT1U")
writeRaster(valley01, file.path(out_dir, "class2_calibrated_valley_no_channel.tif"), overwrite = TRUE, datatype = "INT1U")
writeRaster(upland_low_to_moderate_slope01, file.path(out_dir, "class3_upland_low_to_moderate_slope.tif"), overwrite = TRUE, datatype = "INT1U")
writeRaster(steep_slope01, file.path(out_dir, "class4_steep_slope_gt_threshold.tif"), overwrite = TRUE, datatype = "INT1U")
writeRaster(zones_4, file.path(out_dir, "zones_4class_vectorRiver_calibratedValley_upland_steepSlope.tif"), overwrite = TRUE, datatype = "INT1U")

# -----------------------------
# 15) FINAL SUMMARY TABLES
# -----------------------------
message("Writing summary tables...")

zone_summary <- make_zone_summary_robust(
  zones = zones_4,
  elevation_r = dem,
  slope_deg_r = slope_deg,
  slope_pct_r = slope_pct,
  tpi_large_r = tpi_large,
  dist_to_river_r = dist_to_river_m
)

write.csv(zone_summary, file.path(out_dir, "final_zone_summary.csv"), row.names = FALSE)
write.csv(zone_summary, file.path(out_dir, "final_zone_summary_robust.csv"), row.names = FALSE)
print(zone_summary)

best_exact_metrics <- metrics_from_rasters(best_valley_raw, manual_ref, valid)
best_exact_metrics <- cbind(best[, c("candidate_id", "dist_threshold_m", "slope_threshold_deg", "tpi_threshold_m", "hand_threshold_m", "alluvial_mode")], best_exact_metrics)
write.csv(best_exact_metrics, file.path(out_dir, "best_candidate_exact_metrics.csv"), row.names = FALSE)

# Metadata
metadata <- data.frame(
  parameter = c(
    "dem_file",
    "river_vector_file",
    "manual_valley_polygon_file",
    "alluvial_soil_file",
    "out_dir",
    "resolution_x_m",
    "resolution_y_m",
    "smooth_dem_for_valley",
    "smooth_window_cells",
    "tpi_small_window_m",
    "tpi_large_window_m",
    "channel_buffer_m",
    "steep_slope_threshold_deg",
    "calibration_sample_n",
    "calibration_positive_fraction",
    "exact_top_n",
    "best_candidate_id",
    "best_dist_threshold_m",
    "best_slope_threshold_deg",
    "best_tpi_threshold_m",
    "best_hand_threshold_m",
    "best_alluvial_mode"
  ),
  value = c(
    dem_file,
    river_vector_file,
    manual_valley_polygon_file,
    ifelse(is.na(alluvial_soil_file), "NA", alluvial_soil_file),
    out_dir,
    res_xy[1],
    res_xy[2],
    smooth_dem_for_valley,
    smooth_window_cells,
    tpi_small_window_m,
    tpi_large_window_m,
    channel_buffer_m,
    steep_slope_threshold_deg,
    calibration_sample_n,
    calibration_positive_fraction,
    exact_top_n,
    best$candidate_id,
    best$dist_threshold_m,
    best$slope_threshold_deg,
    best$tpi_threshold_m,
    best$hand_threshold_m,
    best$alluvial_mode
  )
)
write.csv(metadata, file.path(out_dir, "run_metadata.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "R_sessionInfo.txt"))

# -----------------------------
# 16) QUICKLOOK PLOTS
# -----------------------------
message("Writing quicklook plots...")

png(file.path(out_dir, "quicklook_calibrated_valley_workflow.png"), width = 2200, height = 1600, res = 160)
par(mfrow = c(2, 3), mar = c(3, 3, 3, 5))
plot(dem, main = "DEM")
plot(dist_to_river_m, main = "Distance to vector river (m)")
plot(slope_deg, main = "Slope (degrees)")
plot(tpi_large, main = "Large-window TPI")
plot(manual_ref, main = "Manual reference class 2")
plot(zones_4, main = "Final zones: 1 river, 2 valley, 3 upland, 4 steep slope")
dev.off()

png(file.path(out_dir, "quicklook_candidate_ranking.png"), width = 1800, height = 1200, res = 160)
par(mar = c(8, 5, 4, 2))
top_plot <- head(sample_metrics_ranked, 20)
barplot(top_plot$f1,
        names.arg = top_plot$candidate_id,
        las = 2,
        ylab = "Sample F1-score",
        xlab = "Candidate ID",
        main = "Top 20 candidate threshold combinations")
dev.off()

message("Done. Outputs written to: ", normalizePath(out_dir, winslash = "/", mustWork = FALSE))
