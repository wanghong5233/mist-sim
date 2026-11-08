param([switch]$Apply, [switch]$IncludeEngineeringValidation)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$runsRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'runs')).Path
$archivesRoot = Join-Path $projectRoot 'archives/runs'
$pattern = '(?:[A-Za-z0-9_-]+__)?\d{8}T\d{6}_[0-9a-f]{10}'
$protected = [System.Collections.Generic.HashSet[string]]::new()
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'results') -Filter '*.json' -File -Recurse | ForEach-Object {
    foreach ($match in [regex]::Matches([IO.File]::ReadAllText($_.FullName), $pattern)) { [void]$protected.Add($match.Value) }
}
$visited = [System.Collections.Generic.HashSet[string]]::new()
do {
    $pending = @($protected | Where-Object { -not $visited.Contains($_) })
    foreach ($identifier in $pending) {
        [void]$visited.Add($identifier)
        $directory = Join-Path $runsRoot $identifier
        if (Test-Path -LiteralPath $directory) {
            Get-ChildItem -LiteralPath $directory -Filter '*.json' -Recurse -File | ForEach-Object {
                foreach ($match in [regex]::Matches([IO.File]::ReadAllText($_.FullName), $pattern)) { [void]$protected.Add($match.Value) }
            }
        }
    }
} while ($pending.Count -gt 0)
$candidates = @(Get-ChildItem -LiteralPath $runsRoot -Directory | Where-Object {
    $manifestPath = Join-Path $_.FullName 'manifest.json'
    $configPath = Join-Path $_.FullName 'resolved_config.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { return $false }
    $identity = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $engineering = $false
    if ($IncludeEngineeringValidation -and (Test-Path -LiteralPath $configPath)) {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        $engineering = $config.fixture_kind -eq 'synthetic_hand_calculation'
    }
    $engineering -or -not $protected.Contains($identity.run_id)
})
Write-Output "Protected IDs: $($protected.Count); archive candidates: $($candidates.Count)"
if (-not $Apply) { $candidates | Select-Object Name; return }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Directory]::CreateDirectory($archivesRoot) | Out-Null
$reportPath = Join-Path $projectRoot 'results/maintenance/run_archives.csv'
[IO.Directory]::CreateDirectory((Split-Path -Parent $reportPath)) | Out-Null
$rows = @()
if (Test-Path -LiteralPath $reportPath) { $rows = @(Import-Csv -LiteralPath $reportPath) }
foreach ($candidate in $candidates) {
    $resolved = (Resolve-Path -LiteralPath $candidate.FullName).Path
    if (-not $resolved.StartsWith($runsRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside runs' }
    if ($candidate.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point not allowed' }
    $manifestPath = Join-Path $resolved 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { continue }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.status -notin @('completed', 'failed', 'interrupted')) { continue }
    if (@(Get-ChildItem -LiteralPath $resolved -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Nested reparse point' }
    $archivePath = Join-Path $archivesRoot ($candidate.Name + '.zip')
    if (-not (Test-Path -LiteralPath $archivePath)) {
        [IO.Compression.ZipFile]::CreateFromDirectory($resolved, $archivePath, [IO.Compression.CompressionLevel]::Fastest, $false)
    }
    $originals = @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force)
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $entries = @($archive.Entries | Where-Object { $_.Name })
        if ($entries.Count -ne $originals.Count) { throw 'Archive file count differs' }
        foreach ($original in $originals) {
            $relative = $original.FullName.Substring($resolved.Length + 1).Replace('\', '/')
            $entry = $archive.GetEntry($relative)
            if ($null -eq $entry -or $entry.Length -ne $original.Length) { throw "Archive entry mismatch: $relative" }
            $stream = $entry.Open()
            $algorithm = [Security.Cryptography.SHA256]::Create()
            try { $hash = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '') }
            finally { $stream.Dispose(); $algorithm.Dispose() }
            if ($hash -ne (Get-FileHash -LiteralPath $original.FullName -Algorithm SHA256).Hash) { throw "Archive checksum mismatch: $relative" }
        }
    } finally { $archive.Dispose() }
    $rows = @($rows | Where-Object run_id -ne $candidate.Name)
    $rows += [pscustomobject]@{run_id=$candidate.Name; original_path="runs/$($candidate.Name)"; archive_path="archives/runs/$($candidate.Name).zip"; original_bytes=($originals | Measure-Object Length -Sum).Sum; archive_bytes=(Get-Item -LiteralPath $archivePath).Length; archive_sha256=(Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash; files=$originals.Count; verified=$true}
    $temporaryReport = $reportPath + '.partial'
    $rows | Export-Csv -LiteralPath $temporaryReport -NoTypeInformation -Encoding utf8
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try { Move-Item -LiteralPath $temporaryReport -Destination $reportPath -Force; break }
        catch { if ($attempt -eq 4) { throw }; Start-Sleep -Milliseconds 200 }
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
    Write-Output "Archived and verified $($candidate.Name)"
}
