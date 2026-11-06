import Lake
open System Lake DSL

package ffiprobe

input_file sepal_ref.c where
  path := "c" / "sepal_ref.c"
  text := true

target sepal_ref.o pkg : FilePath := do
  let srcJob ← sepal_ref.c.fetch
  let oFile := pkg.buildDir / "c" / "sepal_ref.o"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString]
  buildO oFile srcJob weakArgs #["-fPIC", "-O2", "-ffp-contract=off", "-mfma"] "leanc" getLeanTrace

target libsepalref pkg : FilePath := do
  let o ← sepal_ref.o.fetch
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "sepalref") #[o]

@[default_target]
lean_exe ffitest where
  root := `Main
  moreLinkObjs := #[libsepalref]
