from gostcrypto import gosthash

def stribog_hash(text: str, ver: int) -> bytearray:
    """Хэширует переданную информацию, возвращает в шестнадцатеричном виде."""
    data = text.encode('utf-8')
    
    hash_obj = gosthash.new(f'streebog{ver}')   
    hash_obj.update(data)
    
    digest = hash_obj.digest()
    
    return digest

def last_8_bits_to_decimal(hash_bytes: bytearray) -> int:
    """Берёт последние 8 бит хэша, возвращается десятичное число."""
    last_byte = hash_bytes[-1]
    
    return last_byte

versions = [256, 512]

version = versions[0]
string = 'Куценко Максим Дмитриевич'

hash_result = stribog_hash(string, version)
print(f"Хеш СТРИБОГ-{version}: {hash_result.hex()}")

last_8bits_decimal = last_8_bits_to_decimal(hash_result)
print(f"Последние 8 бит (в десятичном виде): {last_8bits_decimal}")
