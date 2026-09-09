# Собирает цепочку сертификатов сайта api.vk.ru из системного хранилища
# Windows и сохраняет её в certs/ca.pem (PEM).
#
# Нужен, если при запуске бота возникает ошибка:
#   ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
#
# Причина: HTTPS-трафик на машине перехватывает антивирус или прокси со
# своим корневым сертификатом. Python 3.9 не читает системный магазин
# Windows, поэтому такой корень ему нужно передать явно через переменную
# SSL_CA_BUNDLE в .env.
#
# Запуск:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/get_ca.ps1

$ErrorActionPreference = "Stop"

$hostName = "api.vk.ru"
$cb = [System.Net.Security.RemoteCertificateValidationCallback]{
    param($s, $cert, $ch, $errs)
    return $true
}

$client = New-Object System.Net.Sockets.TcpClient($hostName, 443)
$ssl = New-Object System.Net.Security.SslStream($client.GetStream(), $false, $cb)
$ssl.AuthenticateAsClient($hostName)
$leaf = $ssl.RemoteCertificate
$ssl.Dispose()
$client.Dispose()

$leafX = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($leaf)

# Строим полную цепочку (сервер присылает только листовой и промежуточные
# сертификаты, корень берётся из системного хранилища Windows).
$chain = New-Object System.Security.Cryptography.X509Certificates.X509Chain
$ok = $chain.Build($leafX)
Write-Output ("chain build ok: {0}, elements: {1}" -f $ok, $chain.ChainElements.Count)

$outPath = Join-Path (Split-Path $PSScriptRoot) "certs\ca_store.pem"
New-Item -ItemType Directory -Force -Path (Split-Path $outPath) | Out-Null

$sb = New-Object System.Text.StringBuilder
foreach ($el in $chain.ChainElements) {
    $cert = $el.Certificate
    $der = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
    $b64 = [System.Convert]::ToBase64String($der)
    [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
    for ($j = 0; $j -lt $b64.Length; $j += 64) {
        $len = [Math]::Min(64, $b64.Length - $j)
        [void]$sb.AppendLine($b64.Substring($j, $len))
    }
    [void]$sb.AppendLine("-----END CERTIFICATE-----")
}
[System.IO.File]::WriteAllText($outPath, $sb.ToString(), [System.Text.Encoding]::ASCII)
Write-Output ("saved: {0}" -f $outPath)