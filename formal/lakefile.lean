import Lake
open Lake DSL

package "GeoCSV" where
  version := v!"1.0.0"
  keywords := #["ai-safety", "stiefel-manifold", "formal-verification"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

@[default_target]
lean_lib «GeoCSV» where
  roots := #[`GeoCSV.Basic, `GeoCSV.Cayley, `GeoCSV.Kronecker, `GeoCSV.Woodbury]
