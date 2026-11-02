$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "external\XDVioDet"
$outputDirectory = Join-Path $projectRoot "outputs\colab"
$bundlePath = Join-Path $outputDirectory "xd_viodet_rgb_c1_bundle.zip"

$bundleFiles = @(
    @{ Source = "dataset.py"; Entry = "dataset.py" },
    @{ Source = "layers.py"; Entry = "layers.py" },
    @{ Source = "main.py"; Entry = "main.py" },
    @{ Source = "model.py"; Entry = "model.py" },
    @{ Source = "option.py"; Entry = "option.py" },
    @{ Source = "test.py"; Entry = "test.py" },
    @{ Source = "train.py"; Entry = "train.py" },
    @{ Source = "utils.py"; Entry = "utils.py" },
    @{ Source = "README.md"; Entry = "README.md" },
    @{ Source = "list\gt.npy"; Entry = "list/gt.npy" },
    @{ Source = "list\rgb.list"; Entry = "list/rgb.list" },
    @{ Source = "list\rgb_test.list"; Entry = "list/rgb_test.list" }
)

$missingSources = @(
    $bundleFiles |
        Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $sourceRoot $_.Source) -PathType Leaf)
        }
)

if ($missingSources.Count -gt 0) {
    $missingNames = $missingSources.Source -join ", "
    Write-Error "Required XDVioDet bundle inputs are missing: $missingNames"
    exit 1
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$runId = [Guid]::NewGuid().ToString("N")
$stagingDirectory = Join-Path $env:TEMP "xd_viodet_rgb_c1_bundle_$runId"
$temporaryZipPath = Join-Path $outputDirectory ".xd_viodet_rgb_c1_bundle_$runId.tmp.zip"

try {
    New-Item -ItemType Directory -Path $stagingDirectory -Force | Out-Null

    foreach ($file in $bundleFiles) {
        $sourcePath = Join-Path $sourceRoot $file.Source
        $stagedPath = Join-Path $stagingDirectory ($file.Entry.Replace("/", "\"))
        $stagedParent = Split-Path -Parent $stagedPath

        New-Item -ItemType Directory -Path $stagedParent -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $stagedPath
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $stagingDirectory,
        $temporaryZipPath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $false
    )

    $archive = [System.IO.Compression.ZipFile]::OpenRead($temporaryZipPath)
    try {
        $archiveEntries = @{}
        foreach ($entry in $archive.Entries) {
            $normalizedName = $entry.FullName.Replace("\", "/")
            $archiveEntries[$normalizedName] = $true
        }

        $missingEntries = @(
            $bundleFiles |
                Where-Object { -not $archiveEntries.ContainsKey($_.Entry) }
        )

        if ($missingEntries.Count -gt 0) {
            $missingNames = ($missingEntries.Entry) -join ", "
            throw "ZIP verification failed; required entries missing: $missingNames"
        }
    }
    finally {
        $archive.Dispose()
    }

    Move-Item -LiteralPath $temporaryZipPath -Destination $bundlePath -Force
    $bundleSize = (Get-Item -LiteralPath $bundlePath).Length

    Write-Output "Bundle created: $bundlePath"
    Write-Output "Bundle size: $bundleSize bytes"
    Write-Output "Verified required entries: $($bundleFiles.Count)"
}
catch {
    Write-Error "Failed to create Colab bundle: $($_.Exception.Message)"
    exit 1
}
finally {
    if (Test-Path -LiteralPath $stagingDirectory) {
        Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
    }

    if (Test-Path -LiteralPath $temporaryZipPath) {
        Remove-Item -LiteralPath $temporaryZipPath -Force
    }
}