let
  pkgs = import (
    builtins.fetchGit {
      name = "nixos-tensorflow-2";
      url = https://github.com/nixos/nixpkgs;
      ref = "d59b4d07045418bae85a9bdbfdb86d60bc1640bc";}) {};
##  python37 = (let
#    python = let
#      packageOverrides = self: super: {
#        tensorflowWithCuda = super.tensorflowWithCuda.overridePythonAttrs(oldAttrs: {
#          version="2.1.0";
#          src = pkgs.fetchFromGitHub {
#            owner = "tensorflow";
#            repo = "tensorflow";
#            rev = "v2.0.0";
#            sha256 = "1g79xi8yl4sjia8ysk9b7xfzrz83zy28v5dlb2wzmcf0k5pmz60p";
#          };
#          });
#
#      };
#    in pkgs.python37.override {inherit packageOverrides;};
#  in python.withPackages(ps: [ps.tensorflowWithCuda]));
in
  pkgs.mkShell {
    name = "apple";
    buildInputs = [
     pkgs.python37
     pkgs.python37Packages.tensorflowWithCuda
     pkgs.python37Packages.tensorflow-tensorboard
     pkgs.python37Packages.scikitlearn
     pkgs.python37Packages.numba
     pkgs.python37Packages.scipy
     pkgs.python37Packages.pandas
     pkgs.python37Packages.seaborn
     pkgs.python37Packages.h5py
     pkgs.python37Packages.matplotlib
     pkgs.python37Packages.Keras
    ];
    shellHook = ''
      '';

  }
