import pandas as pd
from collections import defaultdict


def extract_restaurant_menus(csv_file_path):
    """
    CSV 파일에서 {영업장명}_{메뉴명} 형태의 컬럼명을 분석하여
    식당별 메뉴 리스트를 추출하는 함수
    """

    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)

    # 식당별 메뉴를 저장할 딕셔너리
    restaurant_menus = defaultdict(list)

    # 컬럼명 분석 (첫 번째 컬럼 '영업일자'는 제외)
    for column in df.columns[1:]:  # 첫 번째 컬럼(영업일자) 제외
        if '_' in column:
            # 언더스코어를 기준으로 식당명과 메뉴명 분리
            parts = column.split('_', 1)  # 첫 번째 언더스코어만 기준으로 분리
            restaurant_name = parts[0]
            menu_name = parts[1]

            # 식당별 메뉴 리스트에 추가
            restaurant_menus[restaurant_name].append(menu_name)

    # 식당명으로 정렬
    sorted_restaurants = dict(sorted(restaurant_menus.items()))

    return sorted_restaurants


def display_restaurant_menus(restaurant_menus):
    """
    식당별 메뉴를 보기 좋게 출력하는 함수
    """
    for restaurant, menus in restaurant_menus.items():
        print(f"{restaurant}: {menus}")
        print()  # 빈 줄 추가


def display_restaurant_menus_formatted(restaurant_menus):
    """
    식당별 메뉴를 더 보기 좋게 포맷팅하여 출력하는 함수
    """
    for restaurant, menus in restaurant_menus.items():
        print(f"■ {restaurant}")
        for i, menu in enumerate(menus, 1):
            print(f"  {i}. {menu}")
        print()  # 빈 줄 추가


# 메인 실행 코드
if __name__ == "__main__":
    # CSV 파일 경로 (실제 파일 경로로 변경하세요)
    csv_file_path = "sample_submission.csv"

    try:
        # 식당별 메뉴 추출
        restaurant_menus = extract_restaurant_menus(csv_file_path)

        print("=== 식당별 메뉴 리스트 ===")
        print()

        # 기본 형태로 출력
        print("1. 기본 형태:")
        display_restaurant_menus(restaurant_menus)

        print("\n" + "=" * 50 + "\n")

        # 포맷팅된 형태로 출력
        print("2. 포맷팅된 형태:")
        display_restaurant_menus_formatted(restaurant_menus)

        # 통계 정보 출력
        print(f"총 식당 수: {len(restaurant_menus)}")
        total_menus = sum(len(menus) for menus in restaurant_menus.values())
        print(f"총 메뉴 수: {total_menus}")

        print("\n=== 식당별 메뉴 개수 ===")
        for restaurant, menus in restaurant_menus.items():
            print(f"{restaurant}: {len(menus)}개")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {csv_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


# 특정 식당의 메뉴만 조회하는 함수
def get_restaurant_menu(restaurant_menus, restaurant_name):
    """
    특정 식당의 메뉴만 반환하는 함수
    """
    if restaurant_name in restaurant_menus:
        return restaurant_menus[restaurant_name]
    else:
        return f"'{restaurant_name}' 식당을 찾을 수 없습니다."

# 사용 예시:
# specific_menu = get_restaurant_menu(restaurant_menus, "담하")
# print(f"담하 메뉴: {specific_menu}")