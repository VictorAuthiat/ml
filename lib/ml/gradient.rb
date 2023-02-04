# frozen_string_literal: true

module ML
  class Gradient
    attr_reader :model_data, :x_data, :y_data, :dw_data, :db_data

    def self.call(model_data, x_data, y_data)
      new(model_data, x_data, y_data).call
    end

    def initialize(model_data, x_data, y_data)
      @model_data = model_data
      @x_data     = x_data
      @y_data     = y_data
      @db_data    = nil
      @dw_data    = nil
    end

    def call
      @dw_data = dot.map { |arr| arr.map { |e| e * y_data_length } }
      @db_data = y_data_length * subtract.flatten.sum

      [dw_data, db_data]
    end

    def y_data_length
      1.0 / y_data.length
    end

    def dot
      transposed_matrix = ML::Utils.transpose_matrix(x_data)

      ML::Utils.dot(transposed_matrix, subtract)
    end

    def subtract
      @_subtract ||= model_data.map.with_index { |arr, i| [arr[0] - y_data[i][0]] }
    end
  end
end
