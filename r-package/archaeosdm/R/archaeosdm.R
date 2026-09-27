#' archaeosdm: R interface to the Python archaeo_sdm toolkit
#'
#' 파이썬 패키지 archaeo_sdm 을 R에서 그대로 쓰기 위한 얇은 래퍼입니다.
#' 계산은 파이썬 쪽에서 수행하고, 결과는 R의 data.frame 과 terra 래스터로 돌려줍니다.
#'
#' 처음 한 번만 파이썬 위치를 알려 주면 됩니다.
#' \code{asdm_python("D:/.../archaeo-sdm/.venv/Scripts/python.exe")}
#'
#' @name archaeosdm
#' @keywords internal
"_PACKAGE"

.asdm <- new.env(parent = emptyenv())

#' 사용할 파이썬 실행 파일과 archaeo_sdm 위치를 지정
#'
#' @param python archaeo-sdm 가상환경의 python.exe 경로
#' @param package_dir archaeo_sdm 패키지가 들어 있는 폴더(보통 python 상위의 프로젝트 폴더)
#' @export
asdm_python <- function(python, package_dir = NULL) {
  reticulate::use_python(python, required = TRUE)
  if (is.null(package_dir)) {
    package_dir <- normalizePath(file.path(dirname(dirname(dirname(python)))),
                                 mustWork = FALSE)
  }
  .asdm$package_dir <- package_dir
  sys <- reticulate::import("sys")
  if (!(package_dir %in% unlist(sys$path))) sys$path$insert(0L, package_dir)
  .asdm$mod <- reticulate::import("archaeo_sdm", delay_load = FALSE)
  invisible(.asdm$mod)
}

.mod <- function() {
  if (is.null(.asdm$mod)) {
    stop("먼저 asdm_python('.../.venv/Scripts/python.exe') 을 호출하세요.", call. = FALSE)
  }
  .asdm$mod
}

#' 파이썬 쪽 준비 상태 확인
#' @export
asdm_available <- function() {
  ok <- !is.null(.asdm$mod)
  if (ok) message("archaeo_sdm ", .mod()$`__version__`, " 준비됨 (", .asdm$package_dir, ")")
  invisible(ok)
}

#' 분석 설정 만들기
#'
#' @param out_dir 결과를 저장할 폴더
#' @param china_db,isomemo_db,ecology_xlsx,point_shapefile,point_csv,gazetteer_manual 자료 경로
#' @param worldclim_dir 현생 WorldClim 폴더
#' @param paleoclim_dir 고기후 폴더(mid/lgm 하위폴더 포함)
#' @param bbox c(서, 남, 동, 북)
#' @param proxies 사용할 동위원소 프록시
#' @param iso_taxa,sdm_taxa 분류군
#' @param ... 그 밖의 '분석' 항목을 덮어쓸 값들
#' @export
asdm_config <- function(out_dir,
                        china_db = "", isomemo_db = "", ecology_xlsx = "",
                        point_shapefile = "", point_csv = "", gazetteer_manual = "",
                        worldclim_dir = "", paleoclim_dir = "",
                        bbox = c(105, 33, 141, 56), bio = c(1, 4, 12, 15),
                        proxies = c("d13C_coll", "d15N_coll"),
                        iso_taxa = c("사람", "돼지"), sdm_taxa = c("돼지"), ...) {
  cfg <- list(
    "출력폴더" = out_dir,
    "자료" = list(china_db = china_db, isomemo_db = isomemo_db,
                  ecology_xlsx = ecology_xlsx, point_shapefile = point_shapefile,
                  point_csv = point_csv, gazetteer_manual = gazetteer_manual),
    "고환경" = list(worldclim_dir = worldclim_dir, paleoclim_dir = paleoclim_dir,
                    bio = as.integer(bio)),
    "범위" = list(bbox = as.numeric(bbox)),
    "분석" = utils::modifyList(
      list("프록시" = proxies,
           "아이소스케이프_분류군" = iso_taxa,
           "SDM_분류군" = sdm_taxa,
           "최소유적수_아이소스케이프" = 3L,
           "최소유적수_SDM" = 4L,
           "지지거리_km" = 350,
           "beta_multiplier" = 3.0,
           "공간블록교차검증" = TRUE),
      list(...))
  )
  structure(cfg, class = c("asdm_config", "list"))
}

#' 전체 분석 실행 (아이소스케이프 + MaxEnt + 시기비교 + 공간 CV)
#'
#' @param config asdm_config() 결과
#' @return 요약 목록. \code{$출력폴더} 안에 GeoTIFF·PNG·CSV가 저장됩니다.
#' @export
asdm_run <- function(config) {
  cli <- reticulate::import("archaeo_sdm.cli")
  res <- cli$run_config(reticulate::r_to_py(unclass(config)))
  out <- reticulate::py_to_r(res)
  message("완료: ", out[["출력폴더"]])
  out
}

#' 표준화된 동위원소 자료표를 R data.frame 으로 읽기
#' @param config asdm_config() 결과
#' @export
asdm_load_dataset <- function(config) {
  cli <- reticulate::import("archaeo_sdm.cli")
  df <- cli$`_load_all`(reticulate::r_to_py(unclass(config)))
  reticulate::py_to_r(df)
}

#' 결과 폴더의 GeoTIFF 들을 terra SpatRaster 로 불러오기
#' @param out_dir 결과 폴더
#' @param pattern 파일 이름 필터 (정규식)
#' @export
asdm_rasters <- function(out_dir, pattern = "\\.tif$") {
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("terra 패키지가 필요합니다: install.packages('terra')", call. = FALSE)
  }
  files <- list.files(out_dir, pattern = pattern, full.names = TRUE)
  if (!length(files)) return(NULL)
  r <- terra::rast(files)
  names(r) <- tools::file_path_sans_ext(basename(files))
  r
}

#' 환경값 행렬로 MaxEnt 직접 학습 (자료를 이미 R에서 준비한 경우)
#'
#' @param presence 출토지점 환경값 행렬 (행=지점, 열=변수)
#' @param background 배경점 환경값 행렬
#' @param var_names 변수 이름
#' @param beta 정규화 계수
#' @return list(model=파이썬 객체, auc=수치, importance=data.frame)
#' @export
asdm_maxent <- function(presence, background, var_names = colnames(presence), beta = 1.0) {
  m <- .mod()$MaxentModel(beta_multiplier = beta)
  m$fit(reticulate::r_to_py(as.matrix(presence)),
        reticulate::r_to_py(as.matrix(background)),
        var_names = as.list(var_names))
  imp <- reticulate::py_to_r(m$permutation_importance(
    reticulate::r_to_py(as.matrix(presence)), reticulate::r_to_py(as.matrix(background))))
  list(model = m,
       auc = m$auc(reticulate::r_to_py(as.matrix(presence)),
                   reticulate::r_to_py(as.matrix(background))),
       importance = data.frame(변수 = names(imp), 중요도 = unlist(imp),
                               row.names = NULL))
}

#' 공간 블록 교차검증
#'
#' @param presence,background 환경값 행렬
#' @param xy_presence,xy_background 좌표 행렬(2열: x, y)
#' @param n_folds 접기 수
#' @param method "kmeans" 또는 "checkerboard"
#' @export
asdm_spatial_cv <- function(presence, background, xy_presence, xy_background,
                            n_folds = 4L, method = "kmeans") {
  v <- reticulate::import("archaeo_sdm.validation")
  res <- v$spatial_block_cv(
    reticulate::r_to_py(as.matrix(presence)),
    reticulate::tuple(xy_presence[, 1], xy_presence[, 2]),
    reticulate::r_to_py(as.matrix(background)),
    reticulate::tuple(xy_background[, 1], xy_background[, 2]),
    n_folds = as.integer(n_folds), method = method)
  out <- reticulate::py_to_r(res)
  out$folds <- as.data.frame(out$folds)
  out
}

#' 고기후 자료 내려받기 (중기 홀로세 'mid' 또는 LGM 'lgm')
#' @param slice "mid" 또는 "lgm"
#' @param out_dir 저장 폴더
#' @export
asdm_download_paleo <- function(slice = c("mid", "lgm"), out_dir) {
  slice <- match.arg(slice)
  pe <- reticulate::import("archaeo_sdm.paleoenv")
  message("내려받는 중… (약 15~18MB)")
  reticulate::py_to_r(pe$download_worldclim_paleo(slice, out_dir, confirm = TRUE))
}
