#!/usr/bin/env bash

help() {
cat <<EOF

Extract library from give jar file

$0 <input jar file> <lib suffix>
  - input jar file: path to jar file with xgboost library
  - lib suffix: suffix to append to xgboost binary library

EOF
}

OUTDIR="target/h2o"
OS=$(uname | sed -e 's/Darwin/osx/' | tr '[:upper:]' '[:lower:]')
BITS=$(getconf LONG_BIT)
PLATFORM="${OS}_${BITS}"

if [ $# -gt 1 ]; then
    jar_file="$1"
    jar_filename=$(basename "$jar_file")
    lib_suffix="_$2"
    # Create output
    rm -rf "${OUTDIR}"
    mkdir -p "${OUTDIR}"

    # Copy jar file
    cp "${jar_file}" "${OUTDIR}"

    # Extract library
    (
        cd  $OUTDIR
        jar -xf "$jar_filename" lib
        # Remove lib from jar file
        echo "Removing native libs from jar file..."
        zip -d "$jar_filename" lib/ 'lib/*'

        # Put library into actual place
        echo "Generating jar file with native libs..."
        mkdir "lib/${PLATFORM}"
        find lib -type f | while read -r f; do
            fname=$(basename "$f")
            fname=${fname//./$lib_suffix.}
            mv "$f" "lib/${PLATFORM}/$fname"
        done
        native_lib_jar=${jar_filename//-/-native-${OS}${lib_suffix}-}
        jar -cf "${native_lib_jar}" ./lib
        rm -rf ./lib
    )


    cat <<EOF

==========
  Please see output in "$(pwd)/target" folder.

$(find "${OUTDIR}" -type f)
==========

EOF
else
    help && exit 1
fi


